# -*- coding: utf-8 -*-
"""
ResidualEmissionsEngine - PACK-021 Net Zero Starter Engine 5
==============================================================

Calculates residual emissions that must be neutralized per the SBTi
Corporate Net-Zero Standard (v1.2, 2024) and provides Carbon Dioxide
Removal (CDR) option assessment with cost estimation.

The SBTi Net-Zero Standard requires companies to reduce value-chain
emissions by at least 90% from base-year levels.  The remaining
residual emissions (up to 10% of base-year total, typically 5-10%
depending on sector) must be neutralized using permanent carbon
dioxide removals (CDR).  Avoidance-only offsets are explicitly
insufficient for neutralization at the net-zero target date.

SBTi Net-Zero Standard Requirements:
    - Section 3: Companies shall set long-term science-based targets
      to reduce Scope 1, 2, and 3 GHG emissions by at least 90%
      before 2050.
    - Section 4: Residual emissions (<=10% of base year) shall be
      neutralized using permanent carbon dioxide removals (CDR).
    - Section 5: Sector-specific residual allowances may permit
      5-10% depending on abatement feasibility.
    - Section 7: Companies should begin procuring CDR capacity well
      in advance of their net-zero target year.

CDR Types Assessed (per IPCC AR6 WGIII, Chapter 12):
    - DACCS (Direct Air Carbon Capture and Storage)
    - BECCS (Bioenergy with Carbon Capture and Storage)
    - Biochar (Biomass pyrolysis for stable carbon)
    - Enhanced Weathering (Accelerated mineral carbonation)
    - Afforestation/Reforestation (Biological sequestration)
    - Ocean-Based CDR (Ocean alkalinity enhancement)
    - Soil Carbon Sequestration (Regenerative agriculture)

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - SBTi Net-Zero Criteria and Recommendations (2024)
    - IPCC AR6 WGIII Chapter 12 - CDR and Mitigation
    - Oxford Principles for Net Zero Aligned Carbon Offsetting (2020)
    - ISO 14068:2023 - Carbon Neutrality
    - EU Carbon Removal Certification Framework (CRCF, 2024)

Zero-Hallucination:
    - Residual budget is arithmetic: base_year * residual_allowance_pct
    - Neutralization requirement = residual budget (deterministic)
    - CDR cost estimation uses fixed reference ranges per CDR type
    - Permanence scoring uses lookup tables (no ML)
    - Timeline projection uses linear year-based planning
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-021 Net Zero Starter
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(
    part: Decimal, whole: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Compute percentage safely: (part / whole) * 100."""
    if whole == Decimal("0"):
        return default
    return part / whole * Decimal("100")


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CDRType(str, Enum):
    """Type of Carbon Dioxide Removal technology or approach.

    Categorizes CDR methods per IPCC AR6 WGIII Chapter 12 and the
    SBTi Net-Zero Standard's guidance on permanent removals.
    """
    DACCS = "daccs"
    BECCS = "beccs"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"
    AFFORESTATION = "afforestation"
    REFORESTATION = "reforestation"
    OCEAN_BASED = "ocean_based"
    SOIL_CARBON = "soil_carbon"


class PermanenceCategory(str, Enum):
    """Permanence classification for CDR storage duration.

    SBTi requires neutralization through permanent removals, defined
    as storage for 100+ years.  Categories align with the EU CRCF
    and Oxford Principles.
    """
    GEOLOGICAL = "geological"
    MINERALOGICAL = "mineralogical"
    BIOLOGICAL_LONG = "biological_long"
    BIOLOGICAL_SHORT = "biological_short"
    OCEAN = "ocean"


class ResidualAllowanceLevel(str, Enum):
    """Residual emission allowance level per SBTi sector guidance.

    Determines the maximum percentage of base-year emissions that
    may remain as residual at the net-zero target date.
    """
    STRICT = "strict"
    STANDARD = "standard"
    ELEVATED = "elevated"


class CDRReadinessLevel(str, Enum):
    """Technology Readiness Level classification for CDR options.

    Based on the IEA/IPCC technology readiness scale adapted for
    CDR deployment context.
    """
    TRL_HIGH = "trl_high"
    TRL_MEDIUM = "trl_medium"
    TRL_LOW = "trl_low"
    TRL_EMERGING = "trl_emerging"


# ---------------------------------------------------------------------------
# Reference Data Constants
# ---------------------------------------------------------------------------


# Sector-specific residual allowance percentages per SBTi guidance.
# Key: sector identifier, Value: max residual % of base-year emissions.
SECTOR_RESIDUAL_ALLOWANCES: Dict[str, Dict[str, Any]] = {
    "energy": {
        "max_residual_pct": Decimal("5.0"),
        "level": ResidualAllowanceLevel.STRICT,
        "rationale": "High abatement potential via electrification and renewables",
    },
    "utilities": {
        "max_residual_pct": Decimal("5.0"),
        "level": ResidualAllowanceLevel.STRICT,
        "rationale": "Grid decarbonization technologies are mature",
    },
    "transportation": {
        "max_residual_pct": Decimal("8.0"),
        "level": ResidualAllowanceLevel.STANDARD,
        "rationale": "Hard-to-abate aviation and shipping subsectors",
    },
    "manufacturing": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Process emissions in cement, steel, chemicals are hard to abate",
    },
    "chemicals": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Feedstock-related and process emissions are difficult to eliminate",
    },
    "cement": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Calcination process emissions have limited abatement options",
    },
    "steel": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Blast furnace reduction emissions are hard to eliminate fully",
    },
    "agriculture": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Biological processes (enteric fermentation, N2O) are inherently difficult",
    },
    "real_estate": {
        "max_residual_pct": Decimal("7.0"),
        "level": ResidualAllowanceLevel.STANDARD,
        "rationale": "Building retrofit potential is high but not complete",
    },
    "financial_services": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Financed emissions depend on portfolio companies' abatement",
    },
    "technology": {
        "max_residual_pct": Decimal("5.0"),
        "level": ResidualAllowanceLevel.STRICT,
        "rationale": "Low direct emissions; Scope 3 supply chain can be addressed",
    },
    "healthcare": {
        "max_residual_pct": Decimal("8.0"),
        "level": ResidualAllowanceLevel.STANDARD,
        "rationale": "Anaesthetic gases and medical waste present challenges",
    },
    "retail": {
        "max_residual_pct": Decimal("7.0"),
        "level": ResidualAllowanceLevel.STANDARD,
        "rationale": "Refrigerant and logistics emissions can be reduced but not eliminated",
    },
    "services": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Scope 3 dominated; dependent on value chain decarbonization",
    },
    "mining": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Fugitive methane and mobile equipment emissions are hard to abate",
    },
    "default": {
        "max_residual_pct": Decimal("10.0"),
        "level": ResidualAllowanceLevel.ELEVATED,
        "rationale": "Default SBTi maximum residual allowance",
    },
}


# CDR option reference data: cost ranges, permanence, and readiness.
# Cost in USD per tCO2e (2024 estimates from literature and market data).
CDR_REFERENCE_DATA: Dict[str, Dict[str, Any]] = {
    CDRType.DACCS.value: {
        "name": "Direct Air Carbon Capture and Storage",
        "cost_low_usd": Decimal("250"),
        "cost_mid_usd": Decimal("450"),
        "cost_high_usd": Decimal("800"),
        "cost_2030_projected_usd": Decimal("200"),
        "cost_2040_projected_usd": Decimal("100"),
        "permanence_years": 10000,
        "permanence_category": PermanenceCategory.GEOLOGICAL,
        "sbti_eligible": True,
        "trl": CDRReadinessLevel.TRL_MEDIUM,
        "trl_score": Decimal("7"),
        "scalability": "High potential, energy-intensive",
        "co_benefits": "Minimal land use, location-flexible",
        "risks": "High energy requirement, cost uncertainty",
        "example_projects": "Climeworks Orca/Mammoth (Iceland), Carbon Engineering (Canada)",
        "annual_capacity_current_mtco2": Decimal("0.01"),
        "annual_capacity_2030_mtco2": Decimal("5"),
        "annual_capacity_2050_mtco2": Decimal("250"),
    },
    CDRType.BECCS.value: {
        "name": "Bioenergy with Carbon Capture and Storage",
        "cost_low_usd": Decimal("100"),
        "cost_mid_usd": Decimal("200"),
        "cost_high_usd": Decimal("350"),
        "cost_2030_projected_usd": Decimal("120"),
        "cost_2040_projected_usd": Decimal("80"),
        "permanence_years": 10000,
        "permanence_category": PermanenceCategory.GEOLOGICAL,
        "sbti_eligible": True,
        "trl": CDRReadinessLevel.TRL_MEDIUM,
        "trl_score": Decimal("6"),
        "scalability": "Moderate, limited by sustainable biomass supply",
        "co_benefits": "Energy generation co-product",
        "risks": "Biomass sustainability, land-use competition",
        "example_projects": "Drax BECCS (UK), Illinois Industrial CCS (US)",
        "annual_capacity_current_mtco2": Decimal("2"),
        "annual_capacity_2030_mtco2": Decimal("20"),
        "annual_capacity_2050_mtco2": Decimal("300"),
    },
    CDRType.BIOCHAR.value: {
        "name": "Biochar Production and Soil Application",
        "cost_low_usd": Decimal("50"),
        "cost_mid_usd": Decimal("120"),
        "cost_high_usd": Decimal("200"),
        "cost_2030_projected_usd": Decimal("80"),
        "cost_2040_projected_usd": Decimal("60"),
        "permanence_years": 500,
        "permanence_category": PermanenceCategory.MINERALOGICAL,
        "sbti_eligible": True,
        "trl": CDRReadinessLevel.TRL_HIGH,
        "trl_score": Decimal("8"),
        "scalability": "Moderate, feedstock-dependent",
        "co_benefits": "Soil fertility improvement, waste management",
        "risks": "Feedstock variability, permanence verification",
        "example_projects": "Carbonfuture (Germany), Pacific Biochar (US)",
        "annual_capacity_current_mtco2": Decimal("2"),
        "annual_capacity_2030_mtco2": Decimal("30"),
        "annual_capacity_2050_mtco2": Decimal("200"),
    },
    CDRType.ENHANCED_WEATHERING.value: {
        "name": "Enhanced Rock Weathering",
        "cost_low_usd": Decimal("50"),
        "cost_mid_usd": Decimal("150"),
        "cost_high_usd": Decimal("300"),
        "cost_2030_projected_usd": Decimal("100"),
        "cost_2040_projected_usd": Decimal("60"),
        "permanence_years": 100000,
        "permanence_category": PermanenceCategory.MINERALOGICAL,
        "sbti_eligible": True,
        "trl": CDRReadinessLevel.TRL_LOW,
        "trl_score": Decimal("5"),
        "scalability": "High potential, mineral supply is abundant",
        "co_benefits": "Soil pH improvement, reduced ocean acidification",
        "risks": "MRV challenges, slow reaction kinetics",
        "example_projects": "UNDO Carbon (UK), Lithos Carbon (US)",
        "annual_capacity_current_mtco2": Decimal("0.1"),
        "annual_capacity_2030_mtco2": Decimal("10"),
        "annual_capacity_2050_mtco2": Decimal("400"),
    },
    CDRType.AFFORESTATION.value: {
        "name": "Afforestation (New Forest Planting)",
        "cost_low_usd": Decimal("10"),
        "cost_mid_usd": Decimal("30"),
        "cost_high_usd": Decimal("80"),
        "cost_2030_projected_usd": Decimal("30"),
        "cost_2040_projected_usd": Decimal("35"),
        "permanence_years": 50,
        "permanence_category": PermanenceCategory.BIOLOGICAL_SHORT,
        "sbti_eligible": False,
        "trl": CDRReadinessLevel.TRL_HIGH,
        "trl_score": Decimal("9"),
        "scalability": "Limited by land availability",
        "co_benefits": "Biodiversity, water regulation, recreation",
        "risks": "Fire, disease, harvesting reversal, land tenure",
        "example_projects": "Various national reforestation programmes",
        "annual_capacity_current_mtco2": Decimal("500"),
        "annual_capacity_2030_mtco2": Decimal("1000"),
        "annual_capacity_2050_mtco2": Decimal("3500"),
    },
    CDRType.REFORESTATION.value: {
        "name": "Reforestation (Restoring Previously Forested Land)",
        "cost_low_usd": Decimal("10"),
        "cost_mid_usd": Decimal("25"),
        "cost_high_usd": Decimal("60"),
        "cost_2030_projected_usd": Decimal("25"),
        "cost_2040_projected_usd": Decimal("30"),
        "permanence_years": 50,
        "permanence_category": PermanenceCategory.BIOLOGICAL_SHORT,
        "sbti_eligible": False,
        "trl": CDRReadinessLevel.TRL_HIGH,
        "trl_score": Decimal("9"),
        "scalability": "Moderate, dependent on available degraded land",
        "co_benefits": "Ecosystem restoration, biodiversity, community benefits",
        "risks": "Fire, disease, harvesting reversal, slow sequestration",
        "example_projects": "Bonn Challenge, national restoration initiatives",
        "annual_capacity_current_mtco2": Decimal("1000"),
        "annual_capacity_2030_mtco2": Decimal("2000"),
        "annual_capacity_2050_mtco2": Decimal("3500"),
    },
    CDRType.OCEAN_BASED.value: {
        "name": "Ocean-Based Carbon Dioxide Removal",
        "cost_low_usd": Decimal("100"),
        "cost_mid_usd": Decimal("300"),
        "cost_high_usd": Decimal("600"),
        "cost_2030_projected_usd": Decimal("200"),
        "cost_2040_projected_usd": Decimal("120"),
        "permanence_years": 1000,
        "permanence_category": PermanenceCategory.OCEAN,
        "sbti_eligible": False,
        "trl": CDRReadinessLevel.TRL_EMERGING,
        "trl_score": Decimal("3"),
        "scalability": "Very high theoretical potential",
        "co_benefits": "Reduced ocean acidification",
        "risks": "Ecosystem impacts, MRV uncertainty, governance gaps",
        "example_projects": "Planetary Technologies (Canada), Running Tide (US)",
        "annual_capacity_current_mtco2": Decimal("0.001"),
        "annual_capacity_2030_mtco2": Decimal("1"),
        "annual_capacity_2050_mtco2": Decimal("100"),
    },
    CDRType.SOIL_CARBON.value: {
        "name": "Soil Carbon Sequestration",
        "cost_low_usd": Decimal("10"),
        "cost_mid_usd": Decimal("40"),
        "cost_high_usd": Decimal("100"),
        "cost_2030_projected_usd": Decimal("35"),
        "cost_2040_projected_usd": Decimal("30"),
        "permanence_years": 30,
        "permanence_category": PermanenceCategory.BIOLOGICAL_SHORT,
        "sbti_eligible": False,
        "trl": CDRReadinessLevel.TRL_HIGH,
        "trl_score": Decimal("8"),
        "scalability": "High, but saturation limits apply",
        "co_benefits": "Improved soil health, agricultural productivity",
        "risks": "Reversibility if practices change, MRV challenges",
        "example_projects": "Indigo Agriculture (US), Soil Capital (EU)",
        "annual_capacity_current_mtco2": Decimal("100"),
        "annual_capacity_2030_mtco2": Decimal("500"),
        "annual_capacity_2050_mtco2": Decimal("2000"),
    },
}

# Permanence category minimum years for SBTi eligibility.
PERMANENCE_THRESHOLDS: Dict[str, int] = {
    "sbti_minimum": 100,
    "high_permanence": 1000,
    "geological": 10000,
}

# Timeline planning defaults.
TIMELINE_DEFAULTS: Dict[str, int] = {
    "procurement_lead_time_years": 3,
    "pilot_phase_years": 2,
    "scale_up_years": 5,
    "full_deployment_buffer_years": 2,
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ResidualInput(BaseModel):
    """Input data for residual emissions calculation.

    Provides the base-year inventory, sector classification, and
    long-term target parameters needed to compute the residual
    budget and neutralization requirements.
    """
    entity_name: str = Field(
        default="",
        description="Name of the reporting entity",
        max_length=300,
    )
    sector: str = Field(
        default="default",
        description="GICS sector or industry classification key",
        max_length=100,
    )
    base_year: int = Field(
        ...,
        description="Base year for the net-zero target",
        ge=1990,
        le=2100,
    )
    base_year_scope1_tco2e: Decimal = Field(
        ...,
        description="Scope 1 emissions in the base year (tCO2e)",
        ge=Decimal("0"),
    )
    base_year_scope2_tco2e: Decimal = Field(
        ...,
        description="Scope 2 emissions in the base year (tCO2e)",
        ge=Decimal("0"),
    )
    base_year_scope3_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Scope 3 emissions in the base year (tCO2e)",
        ge=Decimal("0"),
    )
    target_year: int = Field(
        ...,
        description="Net-zero target year",
        ge=2030,
        le=2100,
    )
    long_term_reduction_pct: Decimal = Field(
        default=Decimal("90.0"),
        description="Long-term reduction target (% of base year, SBTi minimum 90%)",
        ge=Decimal("80"),
        le=Decimal("100"),
    )
    residual_allowance_override_pct: Optional[Decimal] = Field(
        default=None,
        description="Override sector residual allowance (%), if set",
        ge=Decimal("0"),
        le=Decimal("20"),
    )
    current_year: int = Field(
        default=2026,
        description="Current year for timeline projections",
        ge=2020,
        le=2100,
    )
    currency: str = Field(
        default="USD",
        description="Currency for cost estimates",
        max_length=3,
    )
    preferred_cdr_types: List[CDRType] = Field(
        default_factory=list,
        description="Preferred CDR types to prioritize in assessment",
    )

    @field_validator("target_year")
    @classmethod
    def target_after_base(cls, v: int, info: Any) -> int:
        """Validate target year is after base year."""
        base = info.data.get("base_year", 1990)
        if v <= base:
            raise ValueError(
                f"target_year ({v}) must be after base_year ({base})"
            )
        return v


class CDROptionAssessment(BaseModel):
    """Assessment of a single CDR option for neutralization.

    Evaluates the CDR type against cost, permanence, readiness,
    and SBTi eligibility criteria.
    """
    cdr_type: CDRType = Field(
        ..., description="CDR technology type"
    )
    name: str = Field(
        default="", description="Human-readable name"
    )
    cost_low_usd_per_tco2e: Decimal = Field(
        default=Decimal("0"), description="Low-end cost estimate (USD/tCO2e)"
    )
    cost_mid_usd_per_tco2e: Decimal = Field(
        default=Decimal("0"), description="Mid-range cost estimate (USD/tCO2e)"
    )
    cost_high_usd_per_tco2e: Decimal = Field(
        default=Decimal("0"), description="High-end cost estimate (USD/tCO2e)"
    )
    total_cost_low: Decimal = Field(
        default=Decimal("0"), description="Total cost at low estimate"
    )
    total_cost_mid: Decimal = Field(
        default=Decimal("0"), description="Total cost at mid estimate"
    )
    total_cost_high: Decimal = Field(
        default=Decimal("0"), description="Total cost at high estimate"
    )
    permanence_years: int = Field(
        default=0, description="Expected storage permanence (years)"
    )
    permanence_category: PermanenceCategory = Field(
        default=PermanenceCategory.BIOLOGICAL_SHORT,
        description="Permanence classification",
    )
    meets_sbti_permanence: bool = Field(
        default=False,
        description="Whether permanence meets SBTi 100+ year requirement",
    )
    sbti_eligible: bool = Field(
        default=False,
        description="Whether CDR type is SBTi-eligible for neutralization",
    )
    trl: CDRReadinessLevel = Field(
        default=CDRReadinessLevel.TRL_EMERGING,
        description="Technology readiness level",
    )
    trl_score: Decimal = Field(
        default=Decimal("0"), description="TRL numeric score (1-9)"
    )
    scalability_notes: str = Field(
        default="", description="Scalability assessment"
    )
    co_benefits: str = Field(
        default="", description="Environmental and social co-benefits"
    )
    risks: str = Field(
        default="", description="Key risks and uncertainties"
    )
    suitability_score: Decimal = Field(
        default=Decimal("0"),
        description="Weighted suitability score (0-100)",
    )
    recommendation: str = Field(
        default="", description="Recommendation for this CDR option"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class NeutralizationTimeline(BaseModel):
    """Timeline for securing CDR capacity for neutralization.

    Plans the procurement, piloting, and scale-up phases needed
    to have sufficient CDR capacity by the net-zero target year.
    """
    current_year: int = Field(
        default=0, description="Current year"
    )
    target_year: int = Field(
        default=0, description="Net-zero target year"
    )
    years_remaining: int = Field(
        default=0, description="Years until net-zero target"
    )
    pilot_start_year: int = Field(
        default=0, description="Recommended pilot procurement start year"
    )
    scale_up_start_year: int = Field(
        default=0, description="Recommended scale-up start year"
    )
    full_deployment_year: int = Field(
        default=0, description="Full deployment required by year"
    )
    annual_procurement_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Annual CDR procurement needed at full deployment",
    )
    cumulative_procurement_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Cumulative CDR procurement including ramp-up phase",
    )
    milestones: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Timeline milestones with year and action",
    )
    urgency_level: str = Field(
        default="", description="Urgency assessment (low, medium, high, critical)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class ResidualResult(BaseModel):
    """Result of residual emissions and neutralization calculation.

    Contains the residual budget, neutralization requirement, CDR
    option assessments, cost estimates, and procurement timeline.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Calculation timestamp (UTC)",
    )
    entity_name: str = Field(
        default="", description="Reporting entity name"
    )
    sector: str = Field(
        default="", description="Sector classification"
    )
    base_year: int = Field(
        default=0, description="Base year"
    )
    target_year: int = Field(
        default=0, description="Net-zero target year"
    )
    base_year_total_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total base year emissions (S1+S2+S3)",
    )
    base_year_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), description="Base year Scope 1"
    )
    base_year_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), description="Base year Scope 2"
    )
    base_year_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), description="Base year Scope 3"
    )
    long_term_reduction_pct: Decimal = Field(
        default=Decimal("0"), description="Long-term reduction target (%)"
    )
    sector_residual_allowance_pct: Decimal = Field(
        default=Decimal("0"),
        description="Sector-specific residual allowance (%)",
    )
    residual_allowance_level: str = Field(
        default="", description="Residual allowance level label"
    )
    residual_budget_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Maximum residual emissions permitted (tCO2e)",
    )
    neutralization_required_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Neutralization required (tCO2e) = residual budget",
    )
    cdr_options: List[CDROptionAssessment] = Field(
        default_factory=list,
        description="CDR option assessments ranked by suitability",
    )
    sbti_eligible_options_count: int = Field(
        default=0,
        description="Number of SBTi-eligible CDR options",
    )
    total_neutralization_cost_low: Decimal = Field(
        default=Decimal("0"), description="Total cost at low estimate"
    )
    total_neutralization_cost_mid: Decimal = Field(
        default=Decimal("0"), description="Total cost at mid estimate"
    )
    total_neutralization_cost_high: Decimal = Field(
        default=Decimal("0"), description="Total cost at high estimate"
    )
    recommended_cdr_mix: Dict[str, str] = Field(
        default_factory=dict,
        description="Recommended CDR portfolio mix (type -> % allocation)",
    )
    timeline: Optional[NeutralizationTimeline] = Field(
        default=None,
        description="Procurement and deployment timeline",
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings and advisory notes"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Prioritized recommendations"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash of the entire result"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ResidualEmissionsEngine:
    """Residual emissions and neutralization engine per SBTi Net-Zero Standard.

    Provides deterministic, zero-hallucination calculations for:
    - Residual budget from base-year emissions and sector allowance
    - CDR option assessment with cost, permanence, and readiness scoring
    - Neutralization cost estimation across CDR portfolio
    - Procurement timeline planning for CDR capacity
    - SBTi compliance validation for neutralization approach

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Usage::

        engine = ResidualEmissionsEngine()
        result = engine.calculate(ResidualInput(
            entity_name="Acme Corp",
            sector="manufacturing",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("5000"),
            base_year_scope2_tco2e=Decimal("3000"),
            base_year_scope3_tco2e=Decimal("12000"),
            target_year=2050,
        ))
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialize ResidualEmissionsEngine."""
        logger.info(
            "ResidualEmissionsEngine v%s initialized", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Main Calculation                                                     #
    # ------------------------------------------------------------------ #

    def calculate(self, input_data: ResidualInput) -> ResidualResult:
        """Calculate residual emissions budget and neutralization requirements.

        This is the main entry point.  It computes the residual budget,
        assesses CDR options, estimates costs, and builds a timeline.

        Args:
            input_data: Validated ResidualInput with base-year inventory.

        Returns:
            ResidualResult with complete neutralization plan.
        """
        t0 = time.perf_counter()

        logger.info(
            "Calculating residual emissions for '%s' (sector=%s, base=%d, target=%d)",
            input_data.entity_name,
            input_data.sector,
            input_data.base_year,
            input_data.target_year,
        )

        # Step 1: Calculate base-year total
        base_total = self._calculate_base_total(input_data)

        # Step 2: Determine sector residual allowance
        allowance_pct, allowance_level = self._get_residual_allowance(
            input_data.sector, input_data.residual_allowance_override_pct
        )

        # Step 3: Calculate residual budget
        residual_budget = self._calculate_residual_budget(
            base_total, allowance_pct
        )

        # Step 4: Assess CDR options
        cdr_options = self._assess_cdr_options(
            residual_budget, input_data.preferred_cdr_types
        )

        # Step 5: Calculate costs
        cost_low, cost_mid, cost_high = self._calculate_neutralization_costs(
            cdr_options, residual_budget
        )

        # Step 6: Build recommended CDR mix
        recommended_mix = self._build_recommended_mix(cdr_options)

        # Step 7: Build timeline
        timeline = self._build_timeline(
            input_data.current_year,
            input_data.target_year,
            residual_budget,
        )

        # Step 8: Generate warnings and recommendations
        warnings = self._generate_warnings(
            input_data, base_total, allowance_pct, residual_budget
        )
        recommendations = self._generate_recommendations(
            input_data, cdr_options, timeline
        )

        # Count SBTi-eligible options
        sbti_count = sum(1 for o in cdr_options if o.sbti_eligible)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ResidualResult(
            entity_name=input_data.entity_name,
            sector=input_data.sector,
            base_year=input_data.base_year,
            target_year=input_data.target_year,
            base_year_total_tco2e=_round_val(base_total, 3),
            base_year_scope1_tco2e=input_data.base_year_scope1_tco2e,
            base_year_scope2_tco2e=input_data.base_year_scope2_tco2e,
            base_year_scope3_tco2e=input_data.base_year_scope3_tco2e,
            long_term_reduction_pct=input_data.long_term_reduction_pct,
            sector_residual_allowance_pct=allowance_pct,
            residual_allowance_level=allowance_level.value,
            residual_budget_tco2e=_round_val(residual_budget, 3),
            neutralization_required_tco2e=_round_val(residual_budget, 3),
            cdr_options=cdr_options,
            sbti_eligible_options_count=sbti_count,
            total_neutralization_cost_low=_round_val(cost_low, 2),
            total_neutralization_cost_mid=_round_val(cost_mid, 2),
            total_neutralization_cost_high=_round_val(cost_high, 2),
            recommended_cdr_mix=recommended_mix,
            timeline=timeline,
            warnings=warnings,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Residual calculation complete: budget=%.1f tCO2e, "
            "cost_mid=%.0f %s, %d CDR options (%d SBTi-eligible) in %.3f ms",
            float(residual_budget),
            float(cost_mid),
            input_data.currency,
            len(cdr_options),
            sbti_count,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------ #
    # Base-Year Total                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_base_total(self, input_data: ResidualInput) -> Decimal:
        """Calculate total base-year emissions (S1 + S2 + S3).

        Args:
            input_data: Input with scope-level base-year emissions.

        Returns:
            Total base-year emissions in tCO2e.
        """
        total = (
            input_data.base_year_scope1_tco2e
            + input_data.base_year_scope2_tco2e
            + input_data.base_year_scope3_tco2e
        )
        logger.debug(
            "Base year total: S1=%.1f + S2=%.1f + S3=%.1f = %.1f tCO2e",
            float(input_data.base_year_scope1_tco2e),
            float(input_data.base_year_scope2_tco2e),
            float(input_data.base_year_scope3_tco2e),
            float(total),
        )
        return total

    # ------------------------------------------------------------------ #
    # Residual Allowance                                                   #
    # ------------------------------------------------------------------ #

    def _get_residual_allowance(
        self,
        sector: str,
        override_pct: Optional[Decimal],
    ) -> tuple[Decimal, ResidualAllowanceLevel]:
        """Determine the sector-specific residual allowance percentage.

        If an override is provided, it is used directly.  Otherwise,
        the sector lookup table is consulted.

        Args:
            sector: Sector identifier string.
            override_pct: Optional manual override percentage.

        Returns:
            Tuple of (allowance_pct, allowance_level).
        """
        if override_pct is not None:
            logger.info("Using override residual allowance: %.1f%%", float(override_pct))
            if override_pct <= Decimal("5"):
                level = ResidualAllowanceLevel.STRICT
            elif override_pct <= Decimal("8"):
                level = ResidualAllowanceLevel.STANDARD
            else:
                level = ResidualAllowanceLevel.ELEVATED
            return override_pct, level

        sector_key = sector.lower().strip()
        sector_data = SECTOR_RESIDUAL_ALLOWANCES.get(
            sector_key,
            SECTOR_RESIDUAL_ALLOWANCES["default"],
        )

        pct = sector_data["max_residual_pct"]
        level = sector_data["level"]

        logger.info(
            "Sector '%s' residual allowance: %.1f%% (%s)",
            sector_key, float(pct), level.value,
        )
        return pct, level

    # ------------------------------------------------------------------ #
    # Residual Budget                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_residual_budget(
        self, base_total: Decimal, allowance_pct: Decimal
    ) -> Decimal:
        """Calculate the residual emissions budget.

        Formula: residual_budget = base_total * (allowance_pct / 100)

        This is the maximum emissions that may remain at the net-zero
        target date, which must then be neutralized via CDR.

        Args:
            base_total: Total base-year emissions (tCO2e).
            allowance_pct: Residual allowance percentage.

        Returns:
            Residual budget in tCO2e.
        """
        budget = base_total * allowance_pct / Decimal("100")
        logger.info(
            "Residual budget: %.1f * %.1f%% = %.1f tCO2e",
            float(base_total), float(allowance_pct), float(budget),
        )
        return budget

    # ------------------------------------------------------------------ #
    # CDR Option Assessment                                                #
    # ------------------------------------------------------------------ #

    def _assess_cdr_options(
        self,
        neutralization_tco2e: Decimal,
        preferred_types: List[CDRType],
    ) -> List[CDROptionAssessment]:
        """Assess all CDR options against the neutralization requirement.

        Evaluates each CDR type on cost, permanence, SBTi eligibility,
        and technology readiness, producing a ranked list of options.

        Args:
            neutralization_tco2e: Required neutralization volume (tCO2e).
            preferred_types: CDR types to boost in suitability scoring.

        Returns:
            List of CDROptionAssessment sorted by suitability score (desc).
        """
        assessments: List[CDROptionAssessment] = []

        for cdr_key, ref in CDR_REFERENCE_DATA.items():
            cdr_type = CDRType(cdr_key)

            # Calculate total costs for full neutralization volume
            total_low = _round_val(
                neutralization_tco2e * ref["cost_low_usd"], 2
            )
            total_mid = _round_val(
                neutralization_tco2e * ref["cost_mid_usd"], 2
            )
            total_high = _round_val(
                neutralization_tco2e * ref["cost_high_usd"], 2
            )

            # Permanence check against SBTi threshold
            meets_sbti_perm = ref["permanence_years"] >= PERMANENCE_THRESHOLDS["sbti_minimum"]

            # Calculate suitability score (0-100)
            suitability = self._calculate_suitability_score(
                ref, cdr_type, preferred_types
            )

            # Generate recommendation text
            recommendation = self._generate_cdr_recommendation(
                cdr_type, ref, suitability
            )

            assessment = CDROptionAssessment(
                cdr_type=cdr_type,
                name=ref["name"],
                cost_low_usd_per_tco2e=ref["cost_low_usd"],
                cost_mid_usd_per_tco2e=ref["cost_mid_usd"],
                cost_high_usd_per_tco2e=ref["cost_high_usd"],
                total_cost_low=total_low,
                total_cost_mid=total_mid,
                total_cost_high=total_high,
                permanence_years=ref["permanence_years"],
                permanence_category=ref["permanence_category"],
                meets_sbti_permanence=meets_sbti_perm,
                sbti_eligible=ref["sbti_eligible"],
                trl=ref["trl"],
                trl_score=ref["trl_score"],
                scalability_notes=ref["scalability"],
                co_benefits=ref["co_benefits"],
                risks=ref["risks"],
                suitability_score=suitability,
                recommendation=recommendation,
            )
            assessment.provenance_hash = _compute_hash(assessment)
            assessments.append(assessment)

        # Sort by suitability score descending
        assessments.sort(key=lambda a: a.suitability_score, reverse=True)

        logger.info(
            "Assessed %d CDR options, top option: %s (score=%.1f)",
            len(assessments),
            assessments[0].cdr_type.value if assessments else "none",
            float(assessments[0].suitability_score) if assessments else 0,
        )

        return assessments

    def _calculate_suitability_score(
        self,
        ref: Dict[str, Any],
        cdr_type: CDRType,
        preferred_types: List[CDRType],
    ) -> Decimal:
        """Calculate weighted suitability score for a CDR option.

        Scoring dimensions and weights:
            - SBTi eligibility (30%): binary, 100 if eligible, 0 if not
            - Permanence (25%): scaled by years vs geological threshold
            - Technology readiness (20%): TRL score normalized to 100
            - Cost efficiency (15%): inverse of mid cost, normalized
            - Preference bonus (10%): 100 if in preferred list, 0 if not

        Args:
            ref: CDR reference data dictionary.
            cdr_type: The CDR type being scored.
            preferred_types: User-preferred CDR types.

        Returns:
            Suitability score (0-100).
        """
        # SBTi eligibility (30%)
        sbti_score = Decimal("100") if ref["sbti_eligible"] else Decimal("0")

        # Permanence (25%): log-scale normalization
        perm_years = _decimal(ref["permanence_years"])
        perm_threshold = _decimal(PERMANENCE_THRESHOLDS["geological"])
        if perm_years >= perm_threshold:
            perm_score = Decimal("100")
        elif perm_years >= Decimal("100"):
            # Scale 100-10000 years to 50-100
            perm_score = Decimal("50") + _safe_divide(
                (perm_years - Decimal("100")) * Decimal("50"),
                perm_threshold - Decimal("100"),
            )
        elif perm_years >= Decimal("30"):
            # Scale 30-100 years to 10-50
            perm_score = Decimal("10") + _safe_divide(
                (perm_years - Decimal("30")) * Decimal("40"),
                Decimal("70"),
            )
        else:
            perm_score = _safe_divide(
                perm_years * Decimal("10"), Decimal("30")
            )

        # Technology readiness (20%): TRL 1-9 mapped to 0-100
        trl_score = _safe_divide(
            ref["trl_score"] * Decimal("100"), Decimal("9")
        )

        # Cost efficiency (15%): inverse of mid cost, normalized
        # Lower cost = higher score. Cap at 500 USD/tCO2e for normalization.
        mid_cost = ref["cost_mid_usd"]
        cost_cap = Decimal("500")
        if mid_cost <= Decimal("0"):
            cost_score = Decimal("100")
        elif mid_cost >= cost_cap:
            cost_score = Decimal("0")
        else:
            cost_score = (Decimal("1") - mid_cost / cost_cap) * Decimal("100")

        # Preference bonus (10%)
        pref_score = Decimal("100") if cdr_type in preferred_types else Decimal("0")

        # Weighted total
        weighted = (
            sbti_score * Decimal("0.30")
            + perm_score * Decimal("0.25")
            + trl_score * Decimal("0.20")
            + cost_score * Decimal("0.15")
            + pref_score * Decimal("0.10")
        )

        return _round_val(weighted, 1)

    def _generate_cdr_recommendation(
        self,
        cdr_type: CDRType,
        ref: Dict[str, Any],
        suitability: Decimal,
    ) -> str:
        """Generate a recommendation string for a CDR option.

        Args:
            cdr_type: The CDR type.
            ref: Reference data for the CDR type.
            suitability: Calculated suitability score.

        Returns:
            Recommendation string.
        """
        if ref["sbti_eligible"] and suitability >= Decimal("70"):
            return (
                f"Strongly recommended for SBTi neutralization. "
                f"{ref['name']} is SBTi-eligible with {ref['permanence_years']}+ "
                f"years permanence."
            )
        elif ref["sbti_eligible"] and suitability >= Decimal("50"):
            return (
                f"Suitable for SBTi neutralization portfolio. "
                f"Consider as part of a diversified CDR strategy."
            )
        elif ref["sbti_eligible"]:
            return (
                f"SBTi-eligible but may face cost or scalability constraints. "
                f"Monitor technology development and cost trends."
            )
        elif suitability >= Decimal("40"):
            return (
                f"Not SBTi-eligible for net-zero neutralization due to "
                f"insufficient permanence ({ref['permanence_years']} years). "
                f"Suitable for beyond value chain mitigation (BVCM)."
            )
        else:
            return (
                f"Not recommended for SBTi neutralization. "
                f"May be suitable for voluntary compensation or BVCM strategies."
            )

    # ------------------------------------------------------------------ #
    # Cost Estimation                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_neutralization_costs(
        self,
        cdr_options: List[CDROptionAssessment],
        neutralization_tco2e: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Calculate blended neutralization costs across the recommended mix.

        Uses only SBTi-eligible options for cost estimation.  If no
        SBTi-eligible options exist, uses all options as fallback.

        Args:
            cdr_options: List of assessed CDR options.
            neutralization_tco2e: Required neutralization volume.

        Returns:
            Tuple of (cost_low, cost_mid, cost_high) in total.
        """
        eligible = [o for o in cdr_options if o.sbti_eligible]
        if not eligible:
            eligible = cdr_options[:3] if cdr_options else []

        if not eligible:
            return Decimal("0"), Decimal("0"), Decimal("0")

        # Use weighted average based on suitability scores
        total_weight = sum(o.suitability_score for o in eligible)
        if total_weight == Decimal("0"):
            total_weight = _decimal(len(eligible))

        blended_low = Decimal("0")
        blended_mid = Decimal("0")
        blended_high = Decimal("0")

        for option in eligible:
            weight = _safe_divide(option.suitability_score, total_weight)
            blended_low += option.cost_low_usd_per_tco2e * weight
            blended_mid += option.cost_mid_usd_per_tco2e * weight
            blended_high += option.cost_high_usd_per_tco2e * weight

        cost_low = neutralization_tco2e * blended_low
        cost_mid = neutralization_tco2e * blended_mid
        cost_high = neutralization_tco2e * blended_high

        logger.info(
            "Neutralization cost estimate: low=%.0f, mid=%.0f, high=%.0f",
            float(cost_low), float(cost_mid), float(cost_high),
        )

        return cost_low, cost_mid, cost_high

    # ------------------------------------------------------------------ #
    # Recommended CDR Mix                                                  #
    # ------------------------------------------------------------------ #

    def _build_recommended_mix(
        self, cdr_options: List[CDROptionAssessment]
    ) -> Dict[str, str]:
        """Build a recommended CDR portfolio mix allocation.

        Allocates neutralization volume across SBTi-eligible options
        proportionally to their suitability scores.

        Args:
            cdr_options: Assessed CDR options sorted by suitability.

        Returns:
            Dict mapping CDR type to allocation percentage string.
        """
        eligible = [o for o in cdr_options if o.sbti_eligible]
        if not eligible:
            return {}

        total_score = sum(o.suitability_score for o in eligible)
        if total_score == Decimal("0"):
            return {}

        mix: Dict[str, str] = {}
        allocated = Decimal("0")

        for i, option in enumerate(eligible):
            if i == len(eligible) - 1:
                # Last option gets remainder to ensure 100%
                pct = Decimal("100") - allocated
            else:
                pct = _round_val(
                    option.suitability_score / total_score * Decimal("100"), 1
                )
            allocated += pct
            mix[option.cdr_type.value] = str(pct)

        return mix

    # ------------------------------------------------------------------ #
    # Timeline                                                             #
    # ------------------------------------------------------------------ #

    def _build_timeline(
        self,
        current_year: int,
        target_year: int,
        neutralization_tco2e: Decimal,
    ) -> NeutralizationTimeline:
        """Build a procurement and deployment timeline for CDR capacity.

        Plans backward from the target year, allocating time for
        pilot procurement, scale-up, and full deployment.

        Args:
            current_year: Current year.
            target_year: Net-zero target year.
            neutralization_tco2e: Required annual neutralization volume.

        Returns:
            NeutralizationTimeline with milestones and urgency assessment.
        """
        years_remaining = target_year - current_year

        # Calculate key timeline dates
        lead_time = TIMELINE_DEFAULTS["procurement_lead_time_years"]
        pilot_years = TIMELINE_DEFAULTS["pilot_phase_years"]
        scale_years = TIMELINE_DEFAULTS["scale_up_years"]
        buffer_years = TIMELINE_DEFAULTS["full_deployment_buffer_years"]

        full_deployment_year = target_year - buffer_years
        scale_up_start = full_deployment_year - scale_years
        pilot_start = scale_up_start - pilot_years

        # Ensure pilot start is not in the past
        if pilot_start < current_year:
            pilot_start = current_year

        # Annual procurement at full deployment
        annual_procurement = neutralization_tco2e

        # Cumulative procurement during ramp-up
        # Ramp-up: linear increase from pilot to full deployment
        ramp_years = max(full_deployment_year - pilot_start, 1)
        cumulative = Decimal("0")
        for yr_offset in range(ramp_years):
            ramp_fraction = _safe_divide(
                _decimal(yr_offset + 1), _decimal(ramp_years)
            )
            cumulative += neutralization_tco2e * ramp_fraction

        # Milestones
        milestones: List[Dict[str, Any]] = []

        if current_year <= pilot_start:
            milestones.append({
                "year": current_year,
                "action": "Begin CDR market assessment and supplier engagement",
                "phase": "preparation",
            })

        milestones.append({
            "year": pilot_start,
            "action": "Initiate pilot CDR procurement contracts",
            "phase": "pilot",
            "volume_pct": "10-20% of target volume",
        })

        milestones.append({
            "year": min(pilot_start + 1, target_year),
            "action": "Evaluate pilot results and refine CDR portfolio",
            "phase": "pilot",
        })

        milestones.append({
            "year": scale_up_start,
            "action": "Begin scale-up of CDR procurement",
            "phase": "scale_up",
            "volume_pct": "50% of target volume",
        })

        milestones.append({
            "year": min(scale_up_start + 2, target_year),
            "action": "Mid-scale checkpoint: verify CDR delivery and permanence",
            "phase": "scale_up",
        })

        milestones.append({
            "year": full_deployment_year,
            "action": "Full CDR deployment capacity operational",
            "phase": "full_deployment",
            "volume_pct": "100% of target volume",
        })

        milestones.append({
            "year": target_year,
            "action": "Net-zero target year: all residual emissions neutralized",
            "phase": "net_zero",
        })

        # Urgency assessment
        if years_remaining <= 5:
            urgency = "critical"
        elif years_remaining <= 10:
            urgency = "high"
        elif years_remaining <= 20:
            urgency = "medium"
        else:
            urgency = "low"

        timeline = NeutralizationTimeline(
            current_year=current_year,
            target_year=target_year,
            years_remaining=years_remaining,
            pilot_start_year=pilot_start,
            scale_up_start_year=scale_up_start,
            full_deployment_year=full_deployment_year,
            annual_procurement_tco2e=_round_val(annual_procurement, 3),
            cumulative_procurement_tco2e=_round_val(cumulative, 3),
            milestones=milestones,
            urgency_level=urgency,
        )
        timeline.provenance_hash = _compute_hash(timeline)

        logger.info(
            "Timeline built: pilot=%d, scale=%d, deploy=%d, urgency=%s",
            pilot_start, scale_up_start, full_deployment_year, urgency,
        )

        return timeline

    # ------------------------------------------------------------------ #
    # Warnings and Recommendations                                         #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        input_data: ResidualInput,
        base_total: Decimal,
        allowance_pct: Decimal,
        residual_budget: Decimal,
    ) -> List[str]:
        """Generate warnings based on input data and calculations.

        Args:
            input_data: Original input data.
            base_total: Total base-year emissions.
            allowance_pct: Residual allowance percentage.
            residual_budget: Calculated residual budget.

        Returns:
            List of warning strings.
        """
        warnings: List[str] = []

        if input_data.base_year_scope3_tco2e == Decimal("0"):
            warnings.append(
                "Scope 3 emissions are zero. SBTi requires Scope 3 inclusion "
                "if >40% of total. Please verify Scope 3 data completeness."
            )

        if input_data.long_term_reduction_pct < Decimal("90"):
            warnings.append(
                f"Long-term reduction target ({input_data.long_term_reduction_pct}%) "
                f"is below SBTi minimum of 90%. Net-zero claim may not be valid."
            )

        years_to_target = input_data.target_year - input_data.current_year
        if years_to_target < 10:
            warnings.append(
                f"Only {years_to_target} years until net-zero target. "
                f"Immediate action on CDR procurement is required."
            )

        if allowance_pct > Decimal("10"):
            warnings.append(
                f"Residual allowance ({allowance_pct}%) exceeds SBTi maximum "
                f"of 10%. This may not be accepted by SBTi validation."
            )

        # Scope 3 dominance warning
        if base_total > Decimal("0"):
            scope3_share = _safe_pct(
                input_data.base_year_scope3_tco2e, base_total
            )
            if scope3_share > Decimal("80"):
                warnings.append(
                    f"Scope 3 represents {scope3_share}% of total emissions. "
                    f"Residual emissions will be heavily influenced by value "
                    f"chain decarbonization progress."
                )

        return warnings

    def _generate_recommendations(
        self,
        input_data: ResidualInput,
        cdr_options: List[CDROptionAssessment],
        timeline: NeutralizationTimeline,
    ) -> List[str]:
        """Generate prioritized recommendations.

        Args:
            input_data: Original input data.
            cdr_options: Assessed CDR options.
            timeline: Procurement timeline.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Priority 1: SBTi eligibility
        sbti_eligible = [o for o in cdr_options if o.sbti_eligible]
        if sbti_eligible:
            top = sbti_eligible[0]
            recommendations.append(
                f"Prioritize {top.name} as the primary CDR pathway. "
                f"It is SBTi-eligible with {top.permanence_years}+ years "
                f"permanence at {top.cost_mid_usd_per_tco2e} USD/tCO2e."
            )
        else:
            recommendations.append(
                "No SBTi-eligible CDR options are currently preferred. "
                "Consider DACCS, BECCS, biochar, or enhanced weathering "
                "for permanent neutralization."
            )

        # Priority 2: Diversification
        if len(sbti_eligible) >= 2:
            recommendations.append(
                "Diversify CDR portfolio across multiple technologies "
                "to reduce supply risk and cost concentration."
            )

        # Priority 3: Timeline urgency
        if timeline.urgency_level in ("critical", "high"):
            recommendations.append(
                f"CDR procurement urgency is {timeline.urgency_level}. "
                f"Begin pilot procurement immediately (target: "
                f"{timeline.pilot_start_year})."
            )

        # Priority 4: Cost management
        recommendations.append(
            "Secure long-term CDR procurement agreements to lock in "
            "pricing as the voluntary carbon removal market matures."
        )

        # Priority 5: Abatement first
        recommendations.append(
            "Maximize abatement before relying on CDR. SBTi requires "
            "90%+ reduction through direct decarbonization; CDR is "
            "only for genuinely residual emissions."
        )

        return recommendations

    # ------------------------------------------------------------------ #
    # Convenience Methods                                                  #
    # ------------------------------------------------------------------ #

    def get_sector_allowance(self, sector: str) -> Dict[str, Any]:
        """Look up the residual allowance for a given sector.

        Args:
            sector: Sector identifier string.

        Returns:
            Dict with max_residual_pct, level, and rationale.
        """
        sector_key = sector.lower().strip()
        data = SECTOR_RESIDUAL_ALLOWANCES.get(
            sector_key,
            SECTOR_RESIDUAL_ALLOWANCES["default"],
        )
        return {
            "sector": sector_key,
            "max_residual_pct": str(data["max_residual_pct"]),
            "level": data["level"].value,
            "rationale": data["rationale"],
        }

    def get_cdr_info(self, cdr_type: CDRType) -> Dict[str, Any]:
        """Get reference data for a specific CDR type.

        Args:
            cdr_type: CDR technology type.

        Returns:
            Dict with cost ranges, permanence, TRL, and other data.
        """
        ref = CDR_REFERENCE_DATA.get(cdr_type.value, {})
        if not ref:
            return {"error": f"Unknown CDR type: {cdr_type.value}"}

        return {
            "cdr_type": cdr_type.value,
            "name": ref["name"],
            "cost_range_usd_per_tco2e": {
                "low": str(ref["cost_low_usd"]),
                "mid": str(ref["cost_mid_usd"]),
                "high": str(ref["cost_high_usd"]),
            },
            "permanence_years": ref["permanence_years"],
            "permanence_category": ref["permanence_category"].value,
            "sbti_eligible": ref["sbti_eligible"],
            "trl": ref["trl"].value,
            "trl_score": str(ref["trl_score"]),
            "scalability": ref["scalability"],
            "co_benefits": ref["co_benefits"],
            "risks": ref["risks"],
        }

    def list_sbti_eligible_cdr(self) -> List[Dict[str, Any]]:
        """List all CDR types that are SBTi-eligible for neutralization.

        Returns:
            List of dicts with CDR type details for eligible options.
        """
        eligible: List[Dict[str, Any]] = []
        for cdr_key, ref in CDR_REFERENCE_DATA.items():
            if ref["sbti_eligible"]:
                eligible.append({
                    "cdr_type": cdr_key,
                    "name": ref["name"],
                    "permanence_years": ref["permanence_years"],
                    "cost_mid_usd": str(ref["cost_mid_usd"]),
                    "trl_score": str(ref["trl_score"]),
                })
        return eligible
