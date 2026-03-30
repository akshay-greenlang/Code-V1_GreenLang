# -*- coding: utf-8 -*-
"""
FullTaxonomyAlignmentEngine - PACK-011 SFDR Article 9 Engine 3
================================================================

Full EU Taxonomy alignment engine for SFDR Article 9 products per
Taxonomy Regulation Articles 5 and 6, and SFDR RTS Annexes.

Article 9 products that target environmental objectives must disclose the
proportion of investments aligned with the EU Taxonomy across all six
environmental objectives.  The engine computes three-KPI alignment
(turnover, CapEx, OpEx), handles enabling and transitional activities,
applies the gas/nuclear Complementary Delegated Act (CDA), verifies
minimum safeguards (OECD, UNGP, ILO, UDHR), and generates bar chart
data for RTS disclosure templates.

Key Features:
    - Three-KPI alignment: turnover, CapEx, OpEx ratios
    - Six environmental objectives with double-counting prevention
    - Enabling and transitional activity classification
    - Gas/nuclear CDA handling per Regulation (EU) 2022/1214
    - Minimum safeguards verification (Article 18)
    - Article 5 (turnover) and Article 6 (CapEx/OpEx) disclosures
    - Bar chart data generation for RTS Annex III/IV templates

Key Regulatory References:
    - Regulation (EU) 2020/852 (Taxonomy) Articles 3, 5, 6, 9, 10, 16, 17, 18
    - Delegated Regulation (EU) 2021/2139 (Climate Delegated Act)
    - Delegated Regulation (EU) 2022/1214 (Complementary Delegated Act)
    - Delegated Regulation (EU) 2022/1288 (SFDR RTS) Annexes III-V
    - Regulation (EU) 2019/2088 (SFDR) Article 9

Formulas:
    Alignment Ratio = SUM(aligned_holding_value * alignment_pct) / total_nav
    Turnover Ratio  = SUM(w_i * turnover_aligned_pct_i)
    CapEx Ratio     = SUM(w_i * capex_aligned_pct_i)
    OpEx Ratio      = SUM(w_i * opex_aligned_pct_i)
    Enabling Share  = SUM(w_i * enabling_pct_i) / total_aligned
    Transitional    = SUM(w_i * transitional_pct_i) / total_aligned

Zero-Hallucination:
    - All alignment ratios use deterministic Python arithmetic
    - Double-counting prevention via primary objective assignment
    - Minimum safeguards are boolean rule checks, not LLM assessed
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage value or 0.0.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaxonomyEnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives per Article 9."""
    CLIMATE_MITIGATION = "climate_mitigation"
    CLIMATE_ADAPTATION = "climate_adaptation"
    WATER_MARINE = "water_marine"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity"

class ArticleReference(str, Enum):
    """Taxonomy Regulation article references for disclosure."""
    ARTICLE_5 = "article_5"
    ARTICLE_6 = "article_6"
    ARTICLE_5_AND_6 = "article_5_and_6"

class SafeguardArea(str, Enum):
    """Minimum safeguards areas per Article 18 of the Taxonomy Regulation."""
    OECD_MNE_GUIDELINES = "oecd_mne_guidelines"
    UN_GUIDING_PRINCIPLES = "un_guiding_principles"
    ILO_CORE_CONVENTIONS = "ilo_core_conventions"
    UDHR = "universal_declaration_human_rights"
    ANTI_BRIBERY = "anti_bribery"
    TAXATION = "fair_taxation"
    COMPETITION = "fair_competition"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAXONOMY_OBJECTIVES: List[TaxonomyEnvironmentalObjective] = list(
    TaxonomyEnvironmentalObjective
)

# NACE sector codes eligible for gas/nuclear CDA
CDA_GAS_NACE: List[str] = ["D35.11", "D35.30", "C20.11"]
CDA_NUCLEAR_NACE: List[str] = ["D35.11", "E38.12"]

# Maximum enabling/transitional proportions per EU guidance
MAX_TRANSITIONAL_SHARE: float = 100.0
MAX_ENABLING_SHARE: float = 100.0

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class TaxonomyHoldingData(BaseModel):
    """Input data for a single holding with full Taxonomy alignment details.

    Represents turnover, CapEx, and OpEx alignment percentages for each
    of the six environmental objectives, plus enabling/transitional flags,
    gas/nuclear CDA eligibility, and minimum safeguards compliance.

    Attributes:
        holding_id: Unique identifier (ISIN or internal).
        holding_name: Display name of the investee company.
        isin: ISIN code if available.
        sector: NACE or GICS sector code.
        country: Country of domicile (ISO 3166-1 alpha-2).
        nav_value: Net Asset Value of this position in EUR.
        weight_pct: Portfolio weight percentage.
        primary_objective: Primary environmental objective contribution.
        turnover_aligned_pct: Turnover alignment percentage (0-100).
        capex_aligned_pct: CapEx alignment percentage (0-100).
        opex_aligned_pct: OpEx alignment percentage (0-100).
        objective_turnover: Per-objective turnover alignment breakdown.
        objective_capex: Per-objective CapEx alignment breakdown.
        objective_opex: Per-objective OpEx alignment breakdown.
        is_enabling: Whether activity qualifies as enabling (Article 16).
        is_transitional: Whether activity qualifies as transitional (Article 10(2)).
        enabling_pct: Proportion of aligned revenue from enabling activities.
        transitional_pct: Proportion of aligned revenue from transitional activities.
        is_cda_gas: Whether activity falls under gas CDA.
        is_cda_nuclear: Whether activity falls under nuclear CDA.
        cda_gas_turnover_pct: Gas CDA turnover alignment.
        cda_nuclear_turnover_pct: Nuclear CDA turnover alignment.
        minimum_safeguards: Minimum safeguards assessment per area.
        dnsh_passed: Whether DNSH criteria are met.
        substantial_contribution_passed: Whether substantial contribution is met.
        data_source: Source of alignment data.
        reporting_year: Fiscal year of the data.
    """
    holding_id: str = Field(default_factory=_new_uuid, description="Unique holding ID")
    holding_name: str = Field(default="", description="Investee company name")
    isin: str = Field(default="", description="ISIN code")
    sector: str = Field(default="", description="NACE/GICS sector code")
    country: str = Field(default="", description="Country (ISO 3166)")
    nav_value: float = Field(default=0.0, ge=0.0, description="NAV in EUR")
    weight_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Portfolio weight %",
    )

    # Primary objective
    primary_objective: Optional[TaxonomyEnvironmentalObjective] = Field(
        default=None, description="Primary environmental objective",
    )

    # Three-KPI alignment percentages (aggregate)
    turnover_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Turnover alignment %",
    )
    capex_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="CapEx alignment %",
    )
    opex_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="OpEx alignment %",
    )

    # Per-objective breakdowns
    objective_turnover: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-objective turnover alignment breakdown",
    )
    objective_capex: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-objective CapEx alignment breakdown",
    )
    objective_opex: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-objective OpEx alignment breakdown",
    )

    # Enabling / transitional
    is_enabling: bool = Field(
        default=False, description="Enabling activity (Art 16)",
    )
    is_transitional: bool = Field(
        default=False, description="Transitional activity (Art 10(2))",
    )
    enabling_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Share of aligned revenue from enabling activities",
    )
    transitional_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Share of aligned revenue from transitional activities",
    )

    # Gas / nuclear CDA
    is_cda_gas: bool = Field(default=False, description="Gas CDA eligible")
    is_cda_nuclear: bool = Field(
        default=False, description="Nuclear CDA eligible",
    )
    cda_gas_turnover_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Gas CDA turnover alignment %",
    )
    cda_nuclear_turnover_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Nuclear CDA turnover alignment %",
    )

    # Minimum safeguards
    minimum_safeguards: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-area minimum safeguards compliance",
    )

    # Pre-assessed flags
    dnsh_passed: Optional[bool] = Field(
        default=None, description="DNSH criteria met",
    )
    substantial_contribution_passed: Optional[bool] = Field(
        default=None, description="Substantial contribution criteria met",
    )

    # Metadata
    data_source: str = Field(
        default="company_reported", description="Data source",
    )
    reporting_year: int = Field(default=2025, description="Fiscal year of data")

class ObjectiveAlignmentEntry(BaseModel):
    """Alignment metrics for a single environmental objective.

    Provides turnover, CapEx, OpEx ratios plus holding counts
    and contribution to the overall portfolio alignment.
    """
    objective: TaxonomyEnvironmentalObjective = Field(
        description="Environmental objective",
    )
    objective_name: str = Field(
        default="", description="Human-readable objective name",
    )
    holding_count: int = Field(
        default=0, ge=0, description="Holdings contributing",
    )
    turnover_ratio_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Turnover alignment ratio %",
    )
    capex_ratio_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="CapEx alignment ratio %",
    )
    opex_ratio_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="OpEx alignment ratio %",
    )
    nav_aligned: float = Field(
        default=0.0, ge=0.0, description="NAV aligned (EUR)",
    )
    portfolio_share_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Share of total portfolio NAV %",
    )
    enabling_share_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Enabling activity share %",
    )
    transitional_share_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Transitional activity share %",
    )

class MinimumSafeguardsResult(BaseModel):
    """Result of minimum safeguards assessment per Article 18.

    Tracks compliance across OECD MNE Guidelines, UN Guiding Principles,
    ILO Core Conventions, and UDHR for each holding.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID",
    )
    holding_id: str = Field(default="", description="Assessed holding ID")
    holding_name: str = Field(default="", description="Holding name")
    overall_pass: bool = Field(
        default=False, description="All safeguards passed",
    )
    area_results: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-area pass/fail results",
    )
    failed_areas: List[str] = Field(
        default_factory=list,
        description="List of failed safeguard areas",
    )
    data_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of safeguard areas with data",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow, description="Assessment timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class BarChartSeries(BaseModel):
    """A single series (objective) for the RTS bar chart visualization.

    Article 9 products must include a bar chart showing alignment per
    objective for each of the three KPIs.
    """
    objective: TaxonomyEnvironmentalObjective = Field(
        description="Environmental objective",
    )
    label: str = Field(default="", description="Display label")
    turnover_value: float = Field(
        default=0.0, description="Turnover ratio %",
    )
    capex_value: float = Field(
        default=0.0, description="CapEx ratio %",
    )
    opex_value: float = Field(default=0.0, description="OpEx ratio %")
    color_hex: str = Field(
        default="#4CAF50", description="Display color hex code",
    )

class BarChartData(BaseModel):
    """Complete bar chart data for RTS disclosure Annex III/IV.

    Contains all six objective series plus aggregate totals, enabling/
    transitional splits, and gas/nuclear CDA breakdowns.
    """
    chart_id: str = Field(
        default_factory=_new_uuid, description="Chart identifier",
    )
    chart_title: str = Field(
        default="EU Taxonomy Alignment - Environmental Objectives",
        description="Chart title",
    )
    series: List[BarChartSeries] = Field(
        default_factory=list,
        description="Per-objective bar chart series",
    )
    total_turnover_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total turnover alignment %",
    )
    total_capex_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total CapEx alignment %",
    )
    total_opex_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total OpEx alignment %",
    )
    enabling_turnover_pct: float = Field(
        default=0.0, description="Enabling turnover %",
    )
    enabling_capex_pct: float = Field(
        default=0.0, description="Enabling CapEx %",
    )
    enabling_opex_pct: float = Field(
        default=0.0, description="Enabling OpEx %",
    )
    transitional_turnover_pct: float = Field(
        default=0.0, description="Transitional turnover %",
    )
    transitional_capex_pct: float = Field(
        default=0.0, description="Transitional CapEx %",
    )
    transitional_opex_pct: float = Field(
        default=0.0, description="Transitional OpEx %",
    )
    cda_gas_turnover_pct: float = Field(
        default=0.0, description="Gas CDA turnover %",
    )
    cda_nuclear_turnover_pct: float = Field(
        default=0.0, description="Nuclear CDA turnover %",
    )
    non_aligned_pct: float = Field(
        default=0.0, description="Non-aligned %",
    )
    not_eligible_pct: float = Field(
        default=0.0, description="Not Taxonomy-eligible %",
    )
    generated_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class Article5Disclosure(BaseModel):
    """Article 5 (turnover-based) taxonomy disclosure.

    Contains all turnover-based alignment metrics required for the
    pre-contractual and periodic disclosure templates.
    """
    disclosure_id: str = Field(
        default_factory=_new_uuid, description="Disclosure ID",
    )
    product_name: str = Field(
        default="", description="Financial product name",
    )
    reporting_date: datetime = Field(
        default_factory=utcnow, description="Reporting date",
    )
    total_turnover_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total turnover alignment %",
    )
    objective_breakdown: List[ObjectiveAlignmentEntry] = Field(
        default_factory=list,
        description="Per-objective turnover breakdown",
    )
    enabling_share_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Enabling activities share of aligned turnover %",
    )
    transitional_share_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Transitional activities share of aligned turnover %",
    )
    cda_gas_pct: float = Field(
        default=0.0, description="Gas CDA turnover alignment %",
    )
    cda_nuclear_pct: float = Field(
        default=0.0, description="Nuclear CDA turnover alignment %",
    )
    non_aligned_pct: float = Field(
        default=0.0, description="Non-aligned turnover %",
    )
    not_eligible_pct: float = Field(
        default=0.0, description="Not eligible turnover %",
    )
    total_holdings_assessed: int = Field(
        default=0, ge=0, description="Holdings assessed",
    )
    safeguards_pass_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum safeguards pass rate %",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class Article6Disclosure(BaseModel):
    """Article 6 (CapEx/OpEx-based) taxonomy disclosure.

    Contains CapEx and OpEx alignment metrics for disclosure templates,
    particularly useful for companies with forward-looking transition plans.
    """
    disclosure_id: str = Field(
        default_factory=_new_uuid, description="Disclosure ID",
    )
    product_name: str = Field(
        default="", description="Financial product name",
    )
    reporting_date: datetime = Field(
        default_factory=utcnow, description="Reporting date",
    )
    total_capex_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total CapEx alignment %",
    )
    total_opex_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total OpEx alignment %",
    )
    objective_breakdown: List[ObjectiveAlignmentEntry] = Field(
        default_factory=list,
        description="Per-objective CapEx/OpEx breakdown",
    )
    enabling_capex_pct: float = Field(
        default=0.0, description="Enabling CapEx share %",
    )
    enabling_opex_pct: float = Field(
        default=0.0, description="Enabling OpEx share %",
    )
    transitional_capex_pct: float = Field(
        default=0.0, description="Transitional CapEx share %",
    )
    transitional_opex_pct: float = Field(
        default=0.0, description="Transitional OpEx share %",
    )
    non_aligned_capex_pct: float = Field(
        default=0.0, description="Non-aligned CapEx %",
    )
    non_aligned_opex_pct: float = Field(
        default=0.0, description="Non-aligned OpEx %",
    )
    total_holdings_assessed: int = Field(
        default=0, ge=0, description="Holdings assessed",
    )
    safeguards_pass_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum safeguards pass rate %",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class FullTaxonomyResult(BaseModel):
    """Complete result of full Taxonomy alignment assessment.

    Consolidates three-KPI alignment, per-objective breakdowns,
    enabling/transitional splits, gas/nuclear CDA, minimum safeguards,
    Article 5/6 disclosures, and bar chart data.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID",
    )
    product_name: str = Field(
        default="", description="Financial product name",
    )
    reporting_date: datetime = Field(
        default_factory=utcnow, description="Reporting date",
    )

    # Aggregate alignment ratios
    total_turnover_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total turnover alignment %",
    )
    total_capex_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total CapEx alignment %",
    )
    total_opex_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Total OpEx alignment %",
    )

    # Per-objective breakdowns
    objective_breakdown: List[ObjectiveAlignmentEntry] = Field(
        default_factory=list,
        description="Per-objective alignment breakdown",
    )

    # Enabling / transitional
    enabling_turnover_pct: float = Field(
        default=0.0, description="Enabling turnover share %",
    )
    enabling_capex_pct: float = Field(
        default=0.0, description="Enabling CapEx share %",
    )
    enabling_opex_pct: float = Field(
        default=0.0, description="Enabling OpEx share %",
    )
    transitional_turnover_pct: float = Field(
        default=0.0, description="Transitional turnover %",
    )
    transitional_capex_pct: float = Field(
        default=0.0, description="Transitional CapEx %",
    )
    transitional_opex_pct: float = Field(
        default=0.0, description="Transitional OpEx %",
    )

    # Gas / nuclear CDA
    cda_gas_turnover_pct: float = Field(
        default=0.0, description="Gas CDA turnover %",
    )
    cda_gas_capex_pct: float = Field(
        default=0.0, description="Gas CDA CapEx %",
    )
    cda_nuclear_turnover_pct: float = Field(
        default=0.0, description="Nuclear CDA turnover %",
    )
    cda_nuclear_capex_pct: float = Field(
        default=0.0, description="Nuclear CDA CapEx %",
    )

    # Residual
    non_aligned_pct: float = Field(
        default=0.0, description="Non-aligned share %",
    )
    not_eligible_pct: float = Field(
        default=0.0, description="Not eligible share %",
    )

    # Minimum safeguards
    safeguards_results: List[MinimumSafeguardsResult] = Field(
        default_factory=list,
        description="Per-holding safeguards results",
    )
    safeguards_pass_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Overall safeguards pass rate %",
    )

    # Disclosures
    article5_disclosure: Optional[Article5Disclosure] = Field(
        default=None, description="Article 5 (turnover) disclosure",
    )
    article6_disclosure: Optional[Article6Disclosure] = Field(
        default=None, description="Article 6 (CapEx/OpEx) disclosure",
    )

    # Bar chart
    bar_chart_data: Optional[BarChartData] = Field(
        default=None, description="Bar chart data for RTS template",
    )

    # Portfolio summary
    total_nav: float = Field(
        default=0.0, ge=0.0, description="Total portfolio NAV (EUR)",
    )
    total_holdings: int = Field(
        default=0, ge=0, description="Total holdings assessed",
    )
    eligible_holdings: int = Field(
        default=0, ge=0, description="Taxonomy-eligible holdings",
    )
    aligned_holdings: int = Field(
        default=0, ge=0, description="Taxonomy-aligned holdings",
    )

    # Metadata
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class FullTaxonomyConfig(BaseModel):
    """Configuration for the FullTaxonomyAlignmentEngine.

    Controls alignment thresholds, minimum safeguards requirements,
    CDA handling, and disclosure parameters.

    Attributes:
        product_name: Name of the financial product.
        minimum_alignment_pct: Minimum alignment % to consider a holding aligned.
        safeguard_required_areas: Which safeguard areas must pass.
        require_all_safeguards: Whether all safeguard areas must pass.
        enable_cda_gas: Whether to include gas CDA in alignment.
        enable_cda_nuclear: Whether to include nuclear CDA in alignment.
        double_counting_prevention: Whether to prevent double-counting.
        primary_kpi: Primary KPI for overall alignment reporting.
        eligible_threshold_pct: Min weight to consider holding for alignment.
        include_sovereign_bonds: Whether to include sovereign bonds.
    """
    product_name: str = Field(
        default="SFDR Article 9 Product", description="Product name",
    )
    minimum_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum alignment % to consider a holding aligned",
    )
    safeguard_required_areas: List[SafeguardArea] = Field(
        default_factory=lambda: [
            SafeguardArea.OECD_MNE_GUIDELINES,
            SafeguardArea.UN_GUIDING_PRINCIPLES,
            SafeguardArea.ILO_CORE_CONVENTIONS,
            SafeguardArea.UDHR,
        ],
        description="Required minimum safeguard areas",
    )
    require_all_safeguards: bool = Field(
        default=True,
        description="Whether all required safeguard areas must pass",
    )
    enable_cda_gas: bool = Field(
        default=True,
        description="Include gas CDA activities in alignment",
    )
    enable_cda_nuclear: bool = Field(
        default=True,
        description="Include nuclear CDA activities in alignment",
    )
    double_counting_prevention: bool = Field(
        default=True,
        description="Prevent double-counting across objectives",
    )
    primary_kpi: str = Field(
        default="turnover",
        description="Primary KPI for overall alignment (turnover/capex/opex)",
    )
    eligible_threshold_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Min weight to consider holding for alignment",
    )
    include_sovereign_bonds: bool = Field(
        default=False,
        description="Whether to include sovereign bonds in alignment",
    )

# ---------------------------------------------------------------------------
# model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

FullTaxonomyConfig.model_rebuild()
TaxonomyHoldingData.model_rebuild()
ObjectiveAlignmentEntry.model_rebuild()
MinimumSafeguardsResult.model_rebuild()
BarChartSeries.model_rebuild()
BarChartData.model_rebuild()
Article5Disclosure.model_rebuild()
Article6Disclosure.model_rebuild()
FullTaxonomyResult.model_rebuild()

# ---------------------------------------------------------------------------
# Objective color mapping for bar charts
# ---------------------------------------------------------------------------

_OBJECTIVE_COLORS: Dict[TaxonomyEnvironmentalObjective, str] = {
    TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION: "#1B5E20",
    TaxonomyEnvironmentalObjective.CLIMATE_ADAPTATION: "#2E7D32",
    TaxonomyEnvironmentalObjective.WATER_MARINE: "#0277BD",
    TaxonomyEnvironmentalObjective.CIRCULAR_ECONOMY: "#F57F17",
    TaxonomyEnvironmentalObjective.POLLUTION_PREVENTION: "#6A1B9A",
    TaxonomyEnvironmentalObjective.BIODIVERSITY: "#33691E",
}

_OBJECTIVE_LABELS: Dict[TaxonomyEnvironmentalObjective, str] = {
    TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION: "Climate Change Mitigation",
    TaxonomyEnvironmentalObjective.CLIMATE_ADAPTATION: "Climate Change Adaptation",
    TaxonomyEnvironmentalObjective.WATER_MARINE: "Water and Marine Resources",
    TaxonomyEnvironmentalObjective.CIRCULAR_ECONOMY: "Circular Economy",
    TaxonomyEnvironmentalObjective.POLLUTION_PREVENTION: "Pollution Prevention and Control",
    TaxonomyEnvironmentalObjective.BIODIVERSITY: "Biodiversity and Ecosystems",
}

# ---------------------------------------------------------------------------
# FullTaxonomyAlignmentEngine
# ---------------------------------------------------------------------------

class FullTaxonomyAlignmentEngine:
    """
    Full EU Taxonomy alignment engine for SFDR Article 9 products.

    Computes three-KPI alignment ratios (turnover, CapEx, OpEx) across all
    six environmental objectives, handles enabling/transitional classification,
    applies gas/nuclear CDA rules, verifies minimum safeguards per Article 18,
    generates Article 5/6 disclosures, and produces bar chart data for RTS
    templates.

    Zero-Hallucination Guarantees:
        - All alignment ratios are deterministic weighted-average calculations
        - Double-counting prevented via primary objective assignment
        - Minimum safeguards verified by boolean rule checks
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Attributes:
        config: Engine configuration.
        _holdings: Input holding data.
        _total_nav: Calculated total portfolio NAV.
        _safeguards_cache: Cached safeguards results by holding_id.

    Example:
        >>> config = FullTaxonomyConfig(product_name="Green Fund")
        >>> engine = FullTaxonomyAlignmentEngine(config)
        >>> holdings = [TaxonomyHoldingData(
        ...     holding_name="Corp A", nav_value=1e6, weight_pct=10.0,
        ...     turnover_aligned_pct=80.0,
        ...     primary_objective=TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION,
        ... )]
        >>> result = engine.assess_alignment(holdings)
        >>> print(f"Turnover: {result.total_turnover_alignment_pct:.1f}%")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FullTaxonomyAlignmentEngine.

        Args:
            config: Optional configuration dict or FullTaxonomyConfig instance.
        """
        if config and isinstance(config, dict):
            self.config = FullTaxonomyConfig(**config)
        elif config and isinstance(config, FullTaxonomyConfig):
            self.config = config
        else:
            self.config = FullTaxonomyConfig()

        self._holdings: List[TaxonomyHoldingData] = []
        self._total_nav: float = 0.0
        self._safeguards_cache: Dict[str, MinimumSafeguardsResult] = {}

        logger.info(
            "FullTaxonomyAlignmentEngine initialized (version=%s, product=%s)",
            _MODULE_VERSION,
            self.config.product_name,
        )

    # ------------------------------------------------------------------
    # Public API: Full Alignment Assessment
    # ------------------------------------------------------------------

    def assess_alignment(
        self,
        holdings: List[TaxonomyHoldingData],
    ) -> FullTaxonomyResult:
        """Perform comprehensive Taxonomy alignment assessment.

        Computes three-KPI alignment ratios, per-objective breakdowns,
        enabling/transitional splits, CDA adjustments, minimum safeguards,
        Article 5/6 disclosures, and bar chart data.

        Args:
            holdings: List of holding data with Taxonomy alignment details.

        Returns:
            FullTaxonomyResult with complete alignment assessment.

        Raises:
            ValueError: If holdings list is empty.
        """
        start = utcnow()

        if not holdings:
            raise ValueError("Holdings list cannot be empty")

        self._holdings = holdings
        self._total_nav = sum(h.nav_value for h in holdings)
        self._ensure_weights(holdings)

        logger.info(
            "Assessing Taxonomy alignment for %d holdings (NAV=%.2f EUR)",
            len(holdings),
            self._total_nav,
        )

        # Step 1: Verify minimum safeguards for each holding
        safeguards_results = self._assess_all_safeguards(holdings)

        # Step 2: Compute per-objective alignment (double-counting prevention)
        obj_breakdown = self._compute_objective_breakdown(holdings)

        # Step 3: Compute aggregate three-KPI ratios
        turnover_pct = self._compute_weighted_ratio(holdings, "turnover")
        capex_pct = self._compute_weighted_ratio(holdings, "capex")
        opex_pct = self._compute_weighted_ratio(holdings, "opex")

        # Step 4: Compute enabling / transitional shares
        enabling_t, enabling_c, enabling_o = self._compute_activity_type_shares(
            holdings, "enabling"
        )
        trans_t, trans_c, trans_o = self._compute_activity_type_shares(
            holdings, "transitional"
        )

        # Step 5: Compute CDA gas/nuclear shares
        cda_gas_t, cda_gas_c = self._compute_cda_shares(holdings, "gas")
        cda_nuc_t, cda_nuc_c = self._compute_cda_shares(holdings, "nuclear")

        # Step 6: Count eligible and aligned
        eligible_count = sum(
            1 for h in holdings
            if (h.turnover_aligned_pct > 0
                or h.capex_aligned_pct > 0
                or h.opex_aligned_pct > 0
                or h.primary_objective is not None)
        )
        aligned_count = sum(
            1 for h in holdings
            if (h.turnover_aligned_pct > self.config.minimum_alignment_pct
                or h.capex_aligned_pct > self.config.minimum_alignment_pct
                or h.opex_aligned_pct > self.config.minimum_alignment_pct)
        )

        # Step 7: Safeguards pass rate
        safeguards_pass_count = sum(
            1 for s in safeguards_results if s.overall_pass
        )
        safeguards_pass_rate = _safe_pct(
            safeguards_pass_count, len(safeguards_results)
        )

        # Step 8: Residual (non-aligned, not-eligible)
        non_aligned_pct = _round_val(max(0.0, 100.0 - turnover_pct), 4)
        not_eligible_pct = _round_val(
            _safe_pct(
                sum(
                    h.nav_value for h in holdings
                    if h.primary_objective is None
                ),
                self._total_nav,
            ),
            4,
        )

        # Step 9: Build Article 5 disclosure
        article5 = self._build_article5_disclosure(
            holdings, obj_breakdown, turnover_pct,
            enabling_t, trans_t, cda_gas_t, cda_nuc_t,
            safeguards_pass_rate,
        )

        # Step 10: Build Article 6 disclosure
        article6 = self._build_article6_disclosure(
            holdings, obj_breakdown, capex_pct, opex_pct,
            enabling_c, enabling_o, trans_c, trans_o,
            safeguards_pass_rate,
        )

        # Step 11: Build bar chart data
        bar_chart = self._build_bar_chart(
            obj_breakdown, turnover_pct, capex_pct, opex_pct,
            enabling_t, enabling_c, enabling_o,
            trans_t, trans_c, trans_o,
            cda_gas_t, cda_nuc_t, non_aligned_pct, not_eligible_pct,
        )

        processing_ms = (utcnow() - start).total_seconds() * 1000.0

        result = FullTaxonomyResult(
            product_name=self.config.product_name,
            total_turnover_alignment_pct=_round_val(turnover_pct, 4),
            total_capex_alignment_pct=_round_val(capex_pct, 4),
            total_opex_alignment_pct=_round_val(opex_pct, 4),
            objective_breakdown=obj_breakdown,
            enabling_turnover_pct=_round_val(enabling_t, 4),
            enabling_capex_pct=_round_val(enabling_c, 4),
            enabling_opex_pct=_round_val(enabling_o, 4),
            transitional_turnover_pct=_round_val(trans_t, 4),
            transitional_capex_pct=_round_val(trans_c, 4),
            transitional_opex_pct=_round_val(trans_o, 4),
            cda_gas_turnover_pct=_round_val(cda_gas_t, 4),
            cda_gas_capex_pct=_round_val(cda_gas_c, 4),
            cda_nuclear_turnover_pct=_round_val(cda_nuc_t, 4),
            cda_nuclear_capex_pct=_round_val(cda_nuc_c, 4),
            non_aligned_pct=non_aligned_pct,
            not_eligible_pct=not_eligible_pct,
            safeguards_results=safeguards_results,
            safeguards_pass_rate=_round_val(safeguards_pass_rate, 4),
            article5_disclosure=article5,
            article6_disclosure=article6,
            bar_chart_data=bar_chart,
            total_nav=self._total_nav,
            total_holdings=len(holdings),
            eligible_holdings=eligible_count,
            aligned_holdings=aligned_count,
            processing_time_ms=processing_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Taxonomy alignment assessed: turnover=%.2f%%, capex=%.2f%%, "
            "opex=%.2f%%, safeguards=%.1f%%, aligned=%d/%d in %.0fms",
            turnover_pct, capex_pct, opex_pct,
            safeguards_pass_rate, aligned_count, len(holdings),
            processing_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Individual assessments
    # ------------------------------------------------------------------

    def assess_safeguards(
        self,
        holding: TaxonomyHoldingData,
    ) -> MinimumSafeguardsResult:
        """Assess minimum safeguards for a single holding per Article 18.

        Checks compliance with OECD MNE Guidelines, UN Guiding Principles,
        ILO Core Conventions, and UDHR.  All checks are boolean rule-based
        (no LLM involvement).

        Args:
            holding: Holding data with minimum_safeguards assessments.

        Returns:
            MinimumSafeguardsResult with per-area pass/fail.
        """
        area_results: Dict[str, bool] = {}
        failed_areas: List[str] = []
        total_areas = len(self.config.safeguard_required_areas)
        areas_with_data = 0

        for area in self.config.safeguard_required_areas:
            area_value = area.value
            if area_value in holding.minimum_safeguards:
                areas_with_data += 1
                passed = holding.minimum_safeguards[area_value]
                area_results[area_value] = passed
                if not passed:
                    failed_areas.append(area_value)
            else:
                # No data: treat as failed if config requires all safeguards
                area_results[area_value] = not self.config.require_all_safeguards
                if self.config.require_all_safeguards:
                    failed_areas.append(area_value)

        data_coverage = _safe_pct(areas_with_data, total_areas)

        if self.config.require_all_safeguards:
            overall_pass = len(failed_areas) == 0
        else:
            passed_count = sum(1 for v in area_results.values() if v)
            overall_pass = passed_count >= (total_areas // 2 + 1)

        result = MinimumSafeguardsResult(
            holding_id=holding.holding_id,
            holding_name=holding.holding_name,
            overall_pass=overall_pass,
            area_results=area_results,
            failed_areas=failed_areas,
            data_coverage_pct=_round_val(data_coverage, 4),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def compute_turnover_ratio(
        self,
        holdings: List[TaxonomyHoldingData],
    ) -> float:
        """Compute portfolio-level turnover alignment ratio.

        Formula: SUM(nav_i * turnover_aligned_pct_i) / total_nav

        Args:
            holdings: List of holdings with turnover alignment data.

        Returns:
            Turnover alignment ratio as percentage (0-100).
        """
        return self._compute_weighted_ratio(holdings, "turnover")

    def compute_capex_ratio(
        self,
        holdings: List[TaxonomyHoldingData],
    ) -> float:
        """Compute portfolio-level CapEx alignment ratio.

        Formula: SUM(nav_i * capex_aligned_pct_i) / total_nav

        Args:
            holdings: List of holdings with CapEx alignment data.

        Returns:
            CapEx alignment ratio as percentage (0-100).
        """
        return self._compute_weighted_ratio(holdings, "capex")

    def compute_opex_ratio(
        self,
        holdings: List[TaxonomyHoldingData],
    ) -> float:
        """Compute portfolio-level OpEx alignment ratio.

        Formula: SUM(nav_i * opex_aligned_pct_i) / total_nav

        Args:
            holdings: List of holdings with OpEx alignment data.

        Returns:
            OpEx alignment ratio as percentage (0-100).
        """
        return self._compute_weighted_ratio(holdings, "opex")

    def generate_bar_chart_data(
        self,
        holdings: List[TaxonomyHoldingData],
    ) -> BarChartData:
        """Generate bar chart data for RTS Annex III/IV templates.

        Produces per-objective series with turnover, CapEx, and OpEx values
        plus aggregate totals and enabling/transitional splits.

        Args:
            holdings: List of holdings with Taxonomy alignment data.

        Returns:
            BarChartData ready for visualization.
        """
        self._holdings = holdings
        self._total_nav = sum(h.nav_value for h in holdings)
        self._ensure_weights(holdings)

        obj_breakdown = self._compute_objective_breakdown(holdings)
        t_val = self._compute_weighted_ratio(holdings, "turnover")
        c_val = self._compute_weighted_ratio(holdings, "capex")
        o_val = self._compute_weighted_ratio(holdings, "opex")
        e_t, e_c, e_o = self._compute_activity_type_shares(
            holdings, "enabling"
        )
        tr_t, tr_c, tr_o = self._compute_activity_type_shares(
            holdings, "transitional"
        )
        cda_g, _ = self._compute_cda_shares(holdings, "gas")
        cda_n, _ = self._compute_cda_shares(holdings, "nuclear")
        non_aligned = _round_val(max(0.0, 100.0 - t_val), 4)
        not_eligible = _round_val(
            _safe_pct(
                sum(
                    h.nav_value for h in holdings
                    if h.primary_objective is None
                ),
                self._total_nav,
            ),
            4,
        )

        return self._build_bar_chart(
            obj_breakdown, t_val, c_val, o_val,
            e_t, e_c, e_o, tr_t, tr_c, tr_o,
            cda_g, cda_n, non_aligned, not_eligible,
        )

    # ------------------------------------------------------------------
    # Internal: Weight normalization
    # ------------------------------------------------------------------

    def _ensure_weights(
        self, holdings: List[TaxonomyHoldingData],
    ) -> None:
        """Ensure portfolio weights are set (derive from NAV if missing).

        Args:
            holdings: List of holdings to normalize.
        """
        if self._total_nav <= 0.0:
            return
        for h in holdings:
            if h.weight_pct <= 0.0 and h.nav_value > 0.0:
                h.weight_pct = _round_val(
                    (h.nav_value / self._total_nav) * 100.0, 6
                )

    # ------------------------------------------------------------------
    # Internal: Weighted ratio calculation
    # ------------------------------------------------------------------

    def _compute_weighted_ratio(
        self,
        holdings: List[TaxonomyHoldingData],
        kpi: str,
    ) -> float:
        """Compute NAV-weighted alignment ratio for a given KPI.

        Formula: SUM(nav_i * alignment_pct_i) / total_nav

        Args:
            holdings: List of holdings.
            kpi: One of 'turnover', 'capex', 'opex'.

        Returns:
            Portfolio-level alignment ratio (0-100).
        """
        if self._total_nav <= 0.0:
            return 0.0

        total_aligned_nav = 0.0
        for h in holdings:
            if kpi == "turnover":
                pct = h.turnover_aligned_pct
            elif kpi == "capex":
                pct = h.capex_aligned_pct
            else:
                pct = h.opex_aligned_pct

            # Only count if safeguards pass (or not yet checked)
            safeguard_ok = True
            if h.holding_id in self._safeguards_cache:
                safeguard_ok = self._safeguards_cache[
                    h.holding_id
                ].overall_pass

            if safeguard_ok and pct > self.config.minimum_alignment_pct:
                total_aligned_nav += h.nav_value * (pct / 100.0)

        return _round_val(
            _safe_pct(total_aligned_nav, self._total_nav), 4
        )

    # ------------------------------------------------------------------
    # Internal: Per-objective breakdown
    # ------------------------------------------------------------------

    def _compute_objective_breakdown(
        self,
        holdings: List[TaxonomyHoldingData],
    ) -> List[ObjectiveAlignmentEntry]:
        """Compute alignment breakdown by environmental objective.

        Prevents double-counting by assigning each holding to its primary
        objective only (when double_counting_prevention is enabled).

        Args:
            holdings: List of holdings with objective data.

        Returns:
            List of ObjectiveAlignmentEntry, one per objective.
        """
        entries: List[ObjectiveAlignmentEntry] = []

        for obj in TAXONOMY_OBJECTIVES:
            obj_holdings = [
                h for h in holdings if h.primary_objective == obj
            ]

            if not obj_holdings:
                entries.append(ObjectiveAlignmentEntry(
                    objective=obj,
                    objective_name=_OBJECTIVE_LABELS.get(obj, obj.value),
                ))
                continue

            # Compute per-objective weighted ratios
            obj_nav = sum(h.nav_value for h in obj_holdings)
            turnover_sum = sum(
                h.nav_value * (
                    h.objective_turnover.get(
                        obj.value, h.turnover_aligned_pct
                    ) / 100.0
                )
                for h in obj_holdings
            )
            capex_sum = sum(
                h.nav_value * (
                    h.objective_capex.get(
                        obj.value, h.capex_aligned_pct
                    ) / 100.0
                )
                for h in obj_holdings
            )
            opex_sum = sum(
                h.nav_value * (
                    h.objective_opex.get(
                        obj.value, h.opex_aligned_pct
                    ) / 100.0
                )
                for h in obj_holdings
            )

            enabling_nav = sum(
                h.nav_value * (h.enabling_pct / 100.0)
                for h in obj_holdings if h.is_enabling
            )
            transitional_nav = sum(
                h.nav_value * (h.transitional_pct / 100.0)
                for h in obj_holdings if h.is_transitional
            )

            entries.append(ObjectiveAlignmentEntry(
                objective=obj,
                objective_name=_OBJECTIVE_LABELS.get(obj, obj.value),
                holding_count=len(obj_holdings),
                turnover_ratio_pct=_round_val(
                    _safe_pct(turnover_sum, self._total_nav), 4
                ),
                capex_ratio_pct=_round_val(
                    _safe_pct(capex_sum, self._total_nav), 4
                ),
                opex_ratio_pct=_round_val(
                    _safe_pct(opex_sum, self._total_nav), 4
                ),
                nav_aligned=_round_val(obj_nav, 2),
                portfolio_share_pct=_round_val(
                    _safe_pct(obj_nav, self._total_nav), 4
                ),
                enabling_share_pct=_round_val(
                    _safe_pct(enabling_nav, obj_nav), 4
                ),
                transitional_share_pct=_round_val(
                    _safe_pct(transitional_nav, obj_nav), 4
                ),
            ))

        return entries

    # ------------------------------------------------------------------
    # Internal: Enabling / Transitional shares
    # ------------------------------------------------------------------

    def _compute_activity_type_shares(
        self,
        holdings: List[TaxonomyHoldingData],
        activity_type: str,
    ) -> Tuple[float, float, float]:
        """Compute activity type shares across three KPIs.

        Args:
            holdings: List of holdings.
            activity_type: 'enabling' or 'transitional'.

        Returns:
            Tuple of (turnover_share, capex_share, opex_share) percentages.
        """
        if self._total_nav <= 0.0:
            return 0.0, 0.0, 0.0

        is_flag = (
            "is_enabling"
            if activity_type == "enabling"
            else "is_transitional"
        )
        pct_field = (
            "enabling_pct"
            if activity_type == "enabling"
            else "transitional_pct"
        )

        filtered = [
            h for h in holdings if getattr(h, is_flag, False)
        ]
        if not filtered:
            return 0.0, 0.0, 0.0

        t_sum = sum(
            h.nav_value
            * (getattr(h, pct_field, 0.0) / 100.0)
            * (h.turnover_aligned_pct / 100.0)
            for h in filtered
        )
        c_sum = sum(
            h.nav_value
            * (getattr(h, pct_field, 0.0) / 100.0)
            * (h.capex_aligned_pct / 100.0)
            for h in filtered
        )
        o_sum = sum(
            h.nav_value
            * (getattr(h, pct_field, 0.0) / 100.0)
            * (h.opex_aligned_pct / 100.0)
            for h in filtered
        )

        return (
            _round_val(_safe_pct(t_sum, self._total_nav), 4),
            _round_val(_safe_pct(c_sum, self._total_nav), 4),
            _round_val(_safe_pct(o_sum, self._total_nav), 4),
        )

    # ------------------------------------------------------------------
    # Internal: Gas/Nuclear CDA shares
    # ------------------------------------------------------------------

    def _compute_cda_shares(
        self,
        holdings: List[TaxonomyHoldingData],
        cda_type: str,
    ) -> Tuple[float, float]:
        """Compute gas or nuclear CDA alignment shares.

        Args:
            holdings: List of holdings.
            cda_type: 'gas' or 'nuclear'.

        Returns:
            Tuple of (turnover_share, capex_share) percentages.
        """
        if self._total_nav <= 0.0:
            return 0.0, 0.0

        is_flag = (
            "is_cda_gas" if cda_type == "gas" else "is_cda_nuclear"
        )
        t_field = (
            "cda_gas_turnover_pct"
            if cda_type == "gas"
            else "cda_nuclear_turnover_pct"
        )
        enable_flag = (
            self.config.enable_cda_gas
            if cda_type == "gas"
            else self.config.enable_cda_nuclear
        )

        if not enable_flag:
            return 0.0, 0.0

        filtered = [
            h for h in holdings if getattr(h, is_flag, False)
        ]
        if not filtered:
            return 0.0, 0.0

        t_sum = sum(
            h.nav_value * (getattr(h, t_field, 0.0) / 100.0)
            for h in filtered
        )
        c_sum = sum(
            h.nav_value * (h.capex_aligned_pct / 100.0)
            for h in filtered
            if getattr(h, is_flag, False)
        )

        return (
            _round_val(_safe_pct(t_sum, self._total_nav), 4),
            _round_val(_safe_pct(c_sum, self._total_nav), 4),
        )

    # ------------------------------------------------------------------
    # Internal: Minimum safeguards assessment
    # ------------------------------------------------------------------

    def _assess_all_safeguards(
        self,
        holdings: List[TaxonomyHoldingData],
    ) -> List[MinimumSafeguardsResult]:
        """Assess minimum safeguards for all holdings.

        Caches results for use in alignment calculations.

        Args:
            holdings: List of holdings to assess.

        Returns:
            List of MinimumSafeguardsResult.
        """
        self._safeguards_cache = {}
        results: List[MinimumSafeguardsResult] = []

        for h in holdings:
            result = self.assess_safeguards(h)
            self._safeguards_cache[h.holding_id] = result
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Internal: Article 5 disclosure builder
    # ------------------------------------------------------------------

    def _build_article5_disclosure(
        self,
        holdings: List[TaxonomyHoldingData],
        obj_breakdown: List[ObjectiveAlignmentEntry],
        turnover_pct: float,
        enabling_t: float,
        trans_t: float,
        cda_gas_t: float,
        cda_nuc_t: float,
        safeguards_rate: float,
    ) -> Article5Disclosure:
        """Build Article 5 (turnover-based) disclosure.

        Args:
            holdings: List of holdings.
            obj_breakdown: Per-objective breakdown entries.
            turnover_pct: Total turnover alignment %.
            enabling_t: Enabling turnover share %.
            trans_t: Transitional turnover share %.
            cda_gas_t: Gas CDA turnover %.
            cda_nuc_t: Nuclear CDA turnover %.
            safeguards_rate: Minimum safeguards pass rate.

        Returns:
            Article5Disclosure ready for template.
        """
        non_aligned = _round_val(max(0.0, 100.0 - turnover_pct), 4)
        not_eligible = _round_val(
            _safe_pct(
                sum(
                    h.nav_value for h in holdings
                    if h.primary_objective is None
                ),
                self._total_nav,
            ),
            4,
        )

        disclosure = Article5Disclosure(
            product_name=self.config.product_name,
            total_turnover_alignment_pct=_round_val(turnover_pct, 4),
            objective_breakdown=obj_breakdown,
            enabling_share_pct=_round_val(enabling_t, 4),
            transitional_share_pct=_round_val(trans_t, 4),
            cda_gas_pct=_round_val(cda_gas_t, 4),
            cda_nuclear_pct=_round_val(cda_nuc_t, 4),
            non_aligned_pct=non_aligned,
            not_eligible_pct=not_eligible,
            total_holdings_assessed=len(holdings),
            safeguards_pass_rate=_round_val(safeguards_rate, 4),
        )
        disclosure.provenance_hash = _compute_hash(disclosure)
        return disclosure

    # ------------------------------------------------------------------
    # Internal: Article 6 disclosure builder
    # ------------------------------------------------------------------

    def _build_article6_disclosure(
        self,
        holdings: List[TaxonomyHoldingData],
        obj_breakdown: List[ObjectiveAlignmentEntry],
        capex_pct: float,
        opex_pct: float,
        enabling_c: float,
        enabling_o: float,
        trans_c: float,
        trans_o: float,
        safeguards_rate: float,
    ) -> Article6Disclosure:
        """Build Article 6 (CapEx/OpEx-based) disclosure.

        Args:
            holdings: List of holdings.
            obj_breakdown: Per-objective breakdown entries.
            capex_pct: Total CapEx alignment %.
            opex_pct: Total OpEx alignment %.
            enabling_c: Enabling CapEx share %.
            enabling_o: Enabling OpEx share %.
            trans_c: Transitional CapEx share %.
            trans_o: Transitional OpEx share %.
            safeguards_rate: Minimum safeguards pass rate.

        Returns:
            Article6Disclosure ready for template.
        """
        non_aligned_capex = _round_val(max(0.0, 100.0 - capex_pct), 4)
        non_aligned_opex = _round_val(max(0.0, 100.0 - opex_pct), 4)

        disclosure = Article6Disclosure(
            product_name=self.config.product_name,
            total_capex_alignment_pct=_round_val(capex_pct, 4),
            total_opex_alignment_pct=_round_val(opex_pct, 4),
            objective_breakdown=obj_breakdown,
            enabling_capex_pct=_round_val(enabling_c, 4),
            enabling_opex_pct=_round_val(enabling_o, 4),
            transitional_capex_pct=_round_val(trans_c, 4),
            transitional_opex_pct=_round_val(trans_o, 4),
            non_aligned_capex_pct=non_aligned_capex,
            non_aligned_opex_pct=non_aligned_opex,
            total_holdings_assessed=len(holdings),
            safeguards_pass_rate=_round_val(safeguards_rate, 4),
        )
        disclosure.provenance_hash = _compute_hash(disclosure)
        return disclosure

    # ------------------------------------------------------------------
    # Internal: Bar chart builder
    # ------------------------------------------------------------------

    def _build_bar_chart(
        self,
        obj_breakdown: List[ObjectiveAlignmentEntry],
        total_t: float,
        total_c: float,
        total_o: float,
        enabling_t: float,
        enabling_c: float,
        enabling_o: float,
        trans_t: float,
        trans_c: float,
        trans_o: float,
        cda_gas_t: float,
        cda_nuc_t: float,
        non_aligned: float,
        not_eligible: float,
    ) -> BarChartData:
        """Build bar chart data for RTS Annex III/IV templates.

        Args:
            obj_breakdown: Per-objective alignment entries.
            total_t: Total turnover ratio.
            total_c: Total CapEx ratio.
            total_o: Total OpEx ratio.
            enabling_t: Enabling turnover share.
            enabling_c: Enabling CapEx share.
            enabling_o: Enabling OpEx share.
            trans_t: Transitional turnover share.
            trans_c: Transitional CapEx share.
            trans_o: Transitional OpEx share.
            cda_gas_t: Gas CDA turnover share.
            cda_nuc_t: Nuclear CDA turnover share.
            non_aligned: Non-aligned share.
            not_eligible: Not-eligible share.

        Returns:
            BarChartData with all series and totals.
        """
        series: List[BarChartSeries] = []

        for entry in obj_breakdown:
            series.append(BarChartSeries(
                objective=entry.objective,
                label=entry.objective_name,
                turnover_value=entry.turnover_ratio_pct,
                capex_value=entry.capex_ratio_pct,
                opex_value=entry.opex_ratio_pct,
                color_hex=_OBJECTIVE_COLORS.get(
                    entry.objective, "#4CAF50"
                ),
            ))

        chart = BarChartData(
            series=series,
            total_turnover_pct=_round_val(total_t, 4),
            total_capex_pct=_round_val(total_c, 4),
            total_opex_pct=_round_val(total_o, 4),
            enabling_turnover_pct=_round_val(enabling_t, 4),
            enabling_capex_pct=_round_val(enabling_c, 4),
            enabling_opex_pct=_round_val(enabling_o, 4),
            transitional_turnover_pct=_round_val(trans_t, 4),
            transitional_capex_pct=_round_val(trans_c, 4),
            transitional_opex_pct=_round_val(trans_o, 4),
            cda_gas_turnover_pct=_round_val(cda_gas_t, 4),
            cda_nuclear_turnover_pct=_round_val(cda_nuc_t, 4),
            non_aligned_pct=non_aligned,
            not_eligible_pct=not_eligible,
        )
        chart.provenance_hash = _compute_hash(chart)
        return chart
