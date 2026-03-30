# -*- coding: utf-8 -*-
"""
ImpactMeasurementEngine - PACK-011 SFDR Article 9 Engine 4
=============================================================

Sustainability impact measurement engine for SFDR Article 9 products.

Article 9 products must demonstrate measurable positive sustainability
impact through quantitative KPIs.  This engine tracks 15 environmental
and 12 social KPIs, maps portfolio impact to all 17 UN Sustainable
Development Goals (SDGs), implements Theory of Change modelling,
assesses investment additionality, and performs year-on-year comparison
analysis.

Key Features:
    - 15 environmental KPIs (GHG avoided, renewable energy, water saved, etc.)
    - 12 social KPIs (jobs created, people served, education access, etc.)
    - SDG mapping across all 17 goals with contribution scoring
    - Theory of Change (ToC) modelling with stage tracking
    - Additionality assessment (what impact would not occur without investment)
    - Year-on-year (YoY) period comparison for progress tracking
    - KPI definition registry with units, categories, and SDG linkage

Key Regulatory References:
    - Regulation (EU) 2019/2088 (SFDR) Article 9, Article 2(17)
    - Delegated Regulation (EU) 2022/1288 (SFDR RTS) Articles 37-41
    - UN Sustainable Development Goals (SDGs) framework
    - Impact Management Project (IMP) five dimensions of impact

Formulas:
    Portfolio KPI = SUM(weight_i * holding_kpi_i) or SUM(attribution_i * kpi_i)
    Attribution Factor = holding_value_i / enterprise_value_i
    YoY Change = (current_value - prior_value) / |prior_value| * 100
    SDG Score = SUM(kpi_contribution_i * kpi_weight_i) for each SDG
    Additionality = f(counterfactual, intentionality, contribution, materiality)

Zero-Hallucination:
    - All KPI aggregations use deterministic Python arithmetic
    - SDG mappings are statically defined (no LLM classification)
    - YoY comparisons are pure arithmetic on paired data points
    - SHA-256 provenance hash on every result
    - No LLM involvement in any numeric calculation path

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

def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning default on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _yoy_change(current: float, prior: float) -> float:
    """Calculate year-on-year percentage change.

    Args:
        current: Current period value.
        prior: Prior period value.

    Returns:
        Percentage change, or 0.0 if prior is zero.
    """
    if prior == 0.0:
        return 0.0
    return ((current - prior) / abs(prior)) * 100.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ImpactCategory(str, Enum):
    """Impact KPI category classification."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"

class SDGGoal(str, Enum):
    """UN Sustainable Development Goals (1-17)."""
    SDG_1 = "sdg_1_no_poverty"
    SDG_2 = "sdg_2_zero_hunger"
    SDG_3 = "sdg_3_good_health"
    SDG_4 = "sdg_4_quality_education"
    SDG_5 = "sdg_5_gender_equality"
    SDG_6 = "sdg_6_clean_water"
    SDG_7 = "sdg_7_affordable_energy"
    SDG_8 = "sdg_8_decent_work"
    SDG_9 = "sdg_9_industry_innovation"
    SDG_10 = "sdg_10_reduced_inequalities"
    SDG_11 = "sdg_11_sustainable_cities"
    SDG_12 = "sdg_12_responsible_consumption"
    SDG_13 = "sdg_13_climate_action"
    SDG_14 = "sdg_14_life_below_water"
    SDG_15 = "sdg_15_life_on_land"
    SDG_16 = "sdg_16_peace_justice"
    SDG_17 = "sdg_17_partnerships"

class ToCStage(str, Enum):
    """Theory of Change stages."""
    INPUT = "input"
    ACTIVITY = "activity"
    OUTPUT = "output"
    OUTCOME = "outcome"
    IMPACT = "impact"

# ---------------------------------------------------------------------------
# KPI Definition Registry
# ---------------------------------------------------------------------------

# Static mapping: KPI ID -> definition
_ENV_KPI_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "env_ghg_avoided": {
        "name": "GHG Emissions Avoided",
        "unit": "tCO2e",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_13],
        "higher_is_better": True,
    },
    "env_renewable_energy": {
        "name": "Renewable Energy Generated",
        "unit": "MWh",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_7, SDGGoal.SDG_13],
        "higher_is_better": True,
    },
    "env_water_saved": {
        "name": "Water Saved or Treated",
        "unit": "m3",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_6, SDGGoal.SDG_14],
        "higher_is_better": True,
    },
    "env_waste_diverted": {
        "name": "Waste Diverted from Landfill",
        "unit": "tonnes",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_12],
        "higher_is_better": True,
    },
    "env_biodiversity_area": {
        "name": "Biodiversity Area Protected",
        "unit": "hectares",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_15, SDGGoal.SDG_14],
        "higher_is_better": True,
    },
    "env_energy_efficiency": {
        "name": "Energy Efficiency Improvement",
        "unit": "%",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_7, SDGGoal.SDG_13],
        "higher_is_better": True,
    },
    "env_pollution_reduced": {
        "name": "Pollution Reduced (Air/Water/Soil)",
        "unit": "tonnes",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_6, SDGGoal.SDG_11],
        "higher_is_better": True,
    },
    "env_circular_economy": {
        "name": "Materials Recycled or Reused",
        "unit": "tonnes",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_12],
        "higher_is_better": True,
    },
    "env_sustainable_land": {
        "name": "Sustainable Land Use Area",
        "unit": "hectares",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_15, SDGGoal.SDG_2],
        "higher_is_better": True,
    },
    "env_clean_transport": {
        "name": "Clean Transport km Enabled",
        "unit": "million_km",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_11, SDGGoal.SDG_13],
        "higher_is_better": True,
    },
    "env_green_buildings": {
        "name": "Green Building Area Certified",
        "unit": "m2",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_11, SDGGoal.SDG_7],
        "higher_is_better": True,
    },
    "env_carbon_sequestered": {
        "name": "Carbon Sequestered",
        "unit": "tCO2e",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_13, SDGGoal.SDG_15],
        "higher_is_better": True,
    },
    "env_ocean_plastic_removed": {
        "name": "Ocean Plastic Removed",
        "unit": "tonnes",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_14],
        "higher_is_better": True,
    },
    "env_hazardous_waste_treated": {
        "name": "Hazardous Waste Safely Treated",
        "unit": "tonnes",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_6, SDGGoal.SDG_12],
        "higher_is_better": True,
    },
    "env_deforestation_avoided": {
        "name": "Deforestation Avoided",
        "unit": "hectares",
        "category": ImpactCategory.ENVIRONMENTAL,
        "sdgs": [SDGGoal.SDG_15, SDGGoal.SDG_13],
        "higher_is_better": True,
    },
}

_SOCIAL_KPI_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "soc_jobs_created": {
        "name": "Jobs Created",
        "unit": "FTE",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_8],
        "higher_is_better": True,
    },
    "soc_people_served": {
        "name": "People Served (Products/Services)",
        "unit": "persons",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_1, SDGGoal.SDG_3],
        "higher_is_better": True,
    },
    "soc_education_access": {
        "name": "Education Access Provided",
        "unit": "persons",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_4],
        "higher_is_better": True,
    },
    "soc_healthcare_access": {
        "name": "Healthcare Access Provided",
        "unit": "persons",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_3],
        "higher_is_better": True,
    },
    "soc_affordable_housing": {
        "name": "Affordable Housing Units",
        "unit": "units",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_11, SDGGoal.SDG_1],
        "higher_is_better": True,
    },
    "soc_gender_diversity": {
        "name": "Women in Leadership Positions",
        "unit": "%",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_5],
        "higher_is_better": True,
    },
    "soc_financial_inclusion": {
        "name": "Financial Inclusion Beneficiaries",
        "unit": "persons",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_1, SDGGoal.SDG_10],
        "higher_is_better": True,
    },
    "soc_clean_water_access": {
        "name": "Clean Water Access Provided",
        "unit": "persons",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_6],
        "higher_is_better": True,
    },
    "soc_food_security": {
        "name": "Food Security Beneficiaries",
        "unit": "persons",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_2],
        "higher_is_better": True,
    },
    "soc_training_hours": {
        "name": "Employee Training Hours",
        "unit": "hours",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_4, SDGGoal.SDG_8],
        "higher_is_better": True,
    },
    "soc_community_investment": {
        "name": "Community Investment",
        "unit": "EUR",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_11, SDGGoal.SDG_17],
        "higher_is_better": True,
    },
    "soc_accident_rate_reduction": {
        "name": "Workplace Accident Rate Reduction",
        "unit": "%",
        "category": ImpactCategory.SOCIAL,
        "sdgs": [SDGGoal.SDG_8, SDGGoal.SDG_3],
        "higher_is_better": True,
    },
}

ALL_KPI_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    **_ENV_KPI_DEFINITIONS,
    **_SOCIAL_KPI_DEFINITIONS,
}

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class KPIDefinition(BaseModel):
    """Definition of a single impact KPI.

    Describes the KPI's identity, unit of measurement, category,
    SDG linkages, and direction of improvement.
    """
    kpi_id: str = Field(description="Unique KPI identifier")
    name: str = Field(default="", description="Human-readable KPI name")
    unit: str = Field(default="", description="Unit of measurement")
    category: ImpactCategory = Field(description="KPI category")
    sdgs: List[SDGGoal] = Field(
        default_factory=list, description="Linked SDG goals",
    )
    higher_is_better: bool = Field(
        default=True,
        description="Whether higher values indicate positive impact",
    )

class ImpactKPI(BaseModel):
    """A single impact KPI measurement for a holding or portfolio.

    Contains the measured value, data quality, attribution, and
    optional prior-period value for YoY comparison.

    Attributes:
        kpi_id: KPI identifier from the registry.
        holding_id: Holding this KPI belongs to (empty for portfolio-level).
        value: Current period measured value.
        prior_value: Prior period value (for YoY comparison).
        unit: Unit of measurement.
        category: Environmental or social.
        data_quality: Data quality level (reported/estimated/modeled).
        attribution_factor: Portfolio attribution factor (0-1).
        attributed_value: Value after attribution.
        reporting_year: Fiscal year of measurement.
        source: Data source description.
    """
    kpi_id: str = Field(description="KPI identifier")
    holding_id: str = Field(
        default="", description="Holding ID (empty for portfolio-level)",
    )
    value: float = Field(default=0.0, description="Current period value")
    prior_value: Optional[float] = Field(
        default=None, description="Prior period value",
    )
    unit: str = Field(default="", description="Unit of measurement")
    category: ImpactCategory = Field(
        default=ImpactCategory.ENVIRONMENTAL, description="KPI category",
    )
    data_quality: str = Field(
        default="reported",
        description="Data quality (reported/estimated/modeled)",
    )
    attribution_factor: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Portfolio attribution factor",
    )
    attributed_value: float = Field(
        default=0.0, description="Value after attribution",
    )
    reporting_year: int = Field(
        default=2025, description="Fiscal year of measurement",
    )
    source: str = Field(
        default="company_reported", description="Data source",
    )

    @model_validator(mode="after")
    def _compute_attributed(self) -> "ImpactKPI":
        """Auto-compute attributed value if not set."""
        if self.attributed_value == 0.0 and self.value != 0.0:
            self.attributed_value = self.value * self.attribution_factor
        return self

class KPIUpdate(BaseModel):
    """Update payload for a KPI measurement.

    Used to submit new or updated KPI values for a holding.
    """
    kpi_id: str = Field(description="KPI identifier")
    holding_id: str = Field(description="Target holding ID")
    new_value: float = Field(description="New measured value")
    data_quality: str = Field(
        default="reported", description="Data quality level",
    )
    source: str = Field(
        default="manual_update", description="Update source",
    )
    effective_date: datetime = Field(
        default_factory=utcnow, description="Effective date of update",
    )

class SDGContribution(BaseModel):
    """Contribution assessment for a single SDG goal.

    Aggregates all KPIs that map to this SDG and computes a
    weighted contribution score.
    """
    sdg: SDGGoal = Field(description="SDG goal")
    sdg_name: str = Field(default="", description="Human-readable SDG name")
    sdg_number: int = Field(default=0, ge=1, le=17, description="SDG number")
    contributing_kpis: List[str] = Field(
        default_factory=list, description="KPI IDs contributing to this SDG",
    )
    kpi_count: int = Field(
        default=0, ge=0, description="Number of contributing KPIs",
    )
    contribution_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Weighted contribution score (0-100)",
    )
    total_attributed_value: float = Field(
        default=0.0, description="Sum of attributed KPI values for this SDG",
    )
    coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Data coverage percentage for this SDG",
    )
    yoy_change_pct: float = Field(
        default=0.0, description="Year-on-year change in contribution %",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class TheoryOfChange(BaseModel):
    """Theory of Change model for the portfolio's impact thesis.

    Maps the causal chain from investment inputs to ultimate impact,
    tracking evidence and assumptions at each stage.
    """
    toc_id: str = Field(
        default_factory=_new_uuid, description="Theory of Change ID",
    )
    product_name: str = Field(
        default="", description="Financial product name",
    )
    impact_thesis: str = Field(
        default="", description="High-level impact thesis statement",
    )
    stages: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            ToCStage.INPUT.value: [],
            ToCStage.ACTIVITY.value: [],
            ToCStage.OUTPUT.value: [],
            ToCStage.OUTCOME.value: [],
            ToCStage.IMPACT.value: [],
        },
        description="Stage-by-stage causal chain elements",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Key assumptions in the causal chain",
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="Evidence sources supporting the ToC",
    )
    linked_kpis: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="KPIs linked to each ToC stage",
    )
    linked_sdgs: List[SDGGoal] = Field(
        default_factory=list,
        description="SDGs this ToC contributes to",
    )
    completeness_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="ToC completeness score (0-100)",
    )
    last_reviewed: datetime = Field(
        default_factory=utcnow, description="Last review date",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class AdditionalityResult(BaseModel):
    """Result of investment additionality assessment.

    Determines whether the investment creates impact that would not
    have occurred without it (counterfactual analysis).
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID",
    )
    product_name: str = Field(
        default="", description="Financial product name",
    )
    intentionality_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Intentionality of impact (0-100)",
    )
    contribution_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Contribution to outcomes (0-100)",
    )
    counterfactual_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Counterfactual assessment (0-100)",
    )
    materiality_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Materiality of impact (0-100)",
    )
    overall_additionality_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Overall additionality score (0-100)",
    )
    assessment_methodology: str = Field(
        default="imp_five_dimensions",
        description="Methodology used for assessment",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting additionality claim",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow, description="Assessment timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class PeriodComparison(BaseModel):
    """Year-on-year comparison for a specific KPI.

    Tracks the change between two reporting periods for a single
    KPI, both at holding level and portfolio level.
    """
    comparison_id: str = Field(
        default_factory=_new_uuid, description="Comparison ID",
    )
    kpi_id: str = Field(description="KPI identifier")
    kpi_name: str = Field(default="", description="KPI name")
    current_period: str = Field(default="", description="Current period label")
    prior_period: str = Field(default="", description="Prior period label")
    current_value: float = Field(
        default=0.0, description="Current period value",
    )
    prior_value: float = Field(
        default=0.0, description="Prior period value",
    )
    absolute_change: float = Field(
        default=0.0, description="Absolute change (current - prior)",
    )
    pct_change: float = Field(
        default=0.0, description="Percentage change",
    )
    direction: str = Field(
        default="unchanged",
        description="Direction of change (improved/deteriorated/unchanged)",
    )
    higher_is_better: bool = Field(
        default=True, description="Whether higher values are better",
    )
    on_track: bool = Field(
        default=False,
        description="Whether change is in the desired direction",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class ImpactResult(BaseModel):
    """Complete result of the impact measurement assessment.

    Consolidates all KPI aggregations, SDG contributions, Theory of
    Change, additionality, and period comparisons.
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

    # Aggregated KPIs
    environmental_kpis: List[ImpactKPI] = Field(
        default_factory=list,
        description="Portfolio-level environmental KPI aggregations",
    )
    social_kpis: List[ImpactKPI] = Field(
        default_factory=list,
        description="Portfolio-level social KPI aggregations",
    )
    total_kpis_tracked: int = Field(
        default=0, ge=0, description="Total KPIs tracked",
    )
    env_kpi_count: int = Field(
        default=0, ge=0, description="Environmental KPIs tracked",
    )
    social_kpi_count: int = Field(
        default=0, ge=0, description="Social KPIs tracked",
    )

    # SDG contributions
    sdg_contributions: List[SDGContribution] = Field(
        default_factory=list, description="Per-SDG contribution results",
    )
    primary_sdgs: List[SDGGoal] = Field(
        default_factory=list,
        description="Primary SDGs with highest contribution",
    )
    sdg_coverage_count: int = Field(
        default=0, ge=0, description="Number of SDGs with positive contribution",
    )

    # Theory of Change
    theory_of_change: Optional[TheoryOfChange] = Field(
        default=None, description="Theory of Change model",
    )

    # Additionality
    additionality: Optional[AdditionalityResult] = Field(
        default=None, description="Additionality assessment result",
    )

    # Period comparisons
    period_comparisons: List[PeriodComparison] = Field(
        default_factory=list,
        description="Year-on-year KPI comparisons",
    )
    kpis_improved_count: int = Field(
        default=0, ge=0, description="KPIs that improved YoY",
    )
    kpis_deteriorated_count: int = Field(
        default=0, ge=0, description="KPIs that deteriorated YoY",
    )

    # Data quality
    data_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Overall data coverage %",
    )
    reported_data_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Company-reported data %",
    )

    # Portfolio summary
    total_holdings: int = Field(
        default=0, ge=0, description="Total holdings assessed",
    )
    total_nav: float = Field(
        default=0.0, ge=0.0, description="Total portfolio NAV (EUR)",
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

class ImpactConfig(BaseModel):
    """Configuration for the ImpactMeasurementEngine.

    Controls KPI tracking, SDG mapping, additionality methodology,
    and reporting parameters.

    Attributes:
        product_name: Financial product name.
        tracked_env_kpis: List of environmental KPI IDs to track.
        tracked_social_kpis: List of social KPI IDs to track.
        sdg_contribution_threshold: Minimum score to consider SDG positive.
        additionality_methodology: Methodology for additionality assessment.
        min_data_quality_pct: Minimum data quality threshold.
        attribution_method: Method for attributing impact to portfolio.
        primary_sdg_threshold: Score threshold for primary SDG classification.
        current_period_label: Label for the current reporting period.
        prior_period_label: Label for the prior reporting period.
    """
    product_name: str = Field(
        default="SFDR Article 9 Product", description="Product name",
    )
    tracked_env_kpis: List[str] = Field(
        default_factory=lambda: list(_ENV_KPI_DEFINITIONS.keys()),
        description="Environmental KPI IDs to track",
    )
    tracked_social_kpis: List[str] = Field(
        default_factory=lambda: list(_SOCIAL_KPI_DEFINITIONS.keys()),
        description="Social KPI IDs to track",
    )
    sdg_contribution_threshold: float = Field(
        default=10.0, ge=0.0, le=100.0,
        description="Min score to consider SDG contribution positive",
    )
    additionality_methodology: str = Field(
        default="imp_five_dimensions",
        description="Methodology for additionality assessment",
    )
    min_data_quality_pct: float = Field(
        default=70.0, ge=0.0, le=100.0,
        description="Minimum data quality threshold %",
    )
    attribution_method: str = Field(
        default="enterprise_value",
        description="Impact attribution method (enterprise_value/nav_weight)",
    )
    primary_sdg_threshold: float = Field(
        default=30.0, ge=0.0, le=100.0,
        description="Score threshold for primary SDG classification",
    )
    current_period_label: str = Field(
        default="2025", description="Current reporting period label",
    )
    prior_period_label: str = Field(
        default="2024", description="Prior reporting period label",
    )

# ---------------------------------------------------------------------------
# SDG Names
# ---------------------------------------------------------------------------

_SDG_NAMES: Dict[SDGGoal, str] = {
    SDGGoal.SDG_1: "No Poverty",
    SDGGoal.SDG_2: "Zero Hunger",
    SDGGoal.SDG_3: "Good Health and Well-Being",
    SDGGoal.SDG_4: "Quality Education",
    SDGGoal.SDG_5: "Gender Equality",
    SDGGoal.SDG_6: "Clean Water and Sanitation",
    SDGGoal.SDG_7: "Affordable and Clean Energy",
    SDGGoal.SDG_8: "Decent Work and Economic Growth",
    SDGGoal.SDG_9: "Industry, Innovation and Infrastructure",
    SDGGoal.SDG_10: "Reduced Inequalities",
    SDGGoal.SDG_11: "Sustainable Cities and Communities",
    SDGGoal.SDG_12: "Responsible Consumption and Production",
    SDGGoal.SDG_13: "Climate Action",
    SDGGoal.SDG_14: "Life Below Water",
    SDGGoal.SDG_15: "Life on Land",
    SDGGoal.SDG_16: "Peace, Justice and Strong Institutions",
    SDGGoal.SDG_17: "Partnerships for the Goals",
}

_SDG_NUMBERS: Dict[SDGGoal, int] = {
    sdg: i + 1 for i, sdg in enumerate(SDGGoal)
}

# ---------------------------------------------------------------------------
# model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

ImpactConfig.model_rebuild()
ImpactKPI.model_rebuild()
KPIUpdate.model_rebuild()
KPIDefinition.model_rebuild()
SDGContribution.model_rebuild()
TheoryOfChange.model_rebuild()
AdditionalityResult.model_rebuild()
PeriodComparison.model_rebuild()
ImpactResult.model_rebuild()

# ---------------------------------------------------------------------------
# ImpactMeasurementEngine
# ---------------------------------------------------------------------------

class ImpactMeasurementEngine:
    """
    Sustainability impact measurement engine for SFDR Article 9 products.

    Tracks 15 environmental and 12 social KPIs across portfolio holdings,
    maps impact to all 17 UN SDGs, implements Theory of Change modelling,
    assesses investment additionality, and performs year-on-year comparisons.

    Zero-Hallucination Guarantees:
        - All KPI aggregations use deterministic Python arithmetic
        - SDG mappings are statically defined in the KPI registry
        - YoY comparisons are pure arithmetic on paired data points
        - SHA-256 provenance hash on every result
        - No LLM involvement in any numeric calculation path

    Attributes:
        config: Engine configuration.
        _holding_kpis: KPI data keyed by (holding_id, kpi_id).
        _portfolio_kpis: Aggregated portfolio-level KPIs.
        _toc: Theory of Change model.

    Example:
        >>> config = ImpactConfig(product_name="Impact Fund")
        >>> engine = ImpactMeasurementEngine(config)
        >>> kpis = [ImpactKPI(
        ...     kpi_id="env_ghg_avoided", holding_id="H1",
        ...     value=50000.0, attribution_factor=0.05,
        ... )]
        >>> result = engine.assess_impact(kpis)
        >>> print(f"SDGs covered: {result.sdg_coverage_count}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ImpactMeasurementEngine.

        Args:
            config: Optional configuration dict or ImpactConfig instance.
        """
        if config and isinstance(config, dict):
            self.config = ImpactConfig(**config)
        elif config and isinstance(config, ImpactConfig):
            self.config = config
        else:
            self.config = ImpactConfig()

        self._holding_kpis: Dict[Tuple[str, str], ImpactKPI] = {}
        self._portfolio_kpis: Dict[str, ImpactKPI] = {}
        self._toc: Optional[TheoryOfChange] = None

        logger.info(
            "ImpactMeasurementEngine initialized (version=%s, product=%s, "
            "env_kpis=%d, social_kpis=%d)",
            _MODULE_VERSION,
            self.config.product_name,
            len(self.config.tracked_env_kpis),
            len(self.config.tracked_social_kpis),
        )

    # ------------------------------------------------------------------
    # Public API: Full Impact Assessment
    # ------------------------------------------------------------------

    def assess_impact(
        self,
        kpis: List[ImpactKPI],
        theory_of_change: Optional[TheoryOfChange] = None,
        additionality_inputs: Optional[Dict[str, float]] = None,
    ) -> ImpactResult:
        """Perform comprehensive impact measurement assessment.

        Aggregates holding-level KPIs to portfolio level, maps to SDGs,
        performs YoY comparison, evaluates ToC completeness, and assesses
        additionality.

        Args:
            kpis: List of holding-level KPI measurements.
            theory_of_change: Optional Theory of Change model.
            additionality_inputs: Optional dict with intentionality,
                contribution, counterfactual, materiality scores (0-100).

        Returns:
            ImpactResult with complete impact assessment.

        Raises:
            ValueError: If KPI list is empty.
        """
        start = utcnow()

        if not kpis:
            raise ValueError("KPI list cannot be empty")

        # Store holding KPIs
        self._holding_kpis = {}
        for kpi in kpis:
            key = (kpi.holding_id, kpi.kpi_id)
            self._holding_kpis[key] = kpi

        logger.info(
            "Assessing impact with %d KPI measurements", len(kpis),
        )

        # Step 1: Aggregate to portfolio level
        env_aggregated = self._aggregate_kpis(kpis, ImpactCategory.ENVIRONMENTAL)
        social_aggregated = self._aggregate_kpis(kpis, ImpactCategory.SOCIAL)
        self._portfolio_kpis = {
            k.kpi_id: k for k in env_aggregated + social_aggregated
        }

        # Step 2: Map to SDGs
        sdg_contributions = self._compute_sdg_contributions(
            env_aggregated + social_aggregated
        )

        # Step 3: Identify primary SDGs
        primary_sdgs = [
            s.sdg for s in sdg_contributions
            if s.contribution_score >= self.config.primary_sdg_threshold
        ]

        # Step 4: Perform YoY comparison
        comparisons = self._compute_period_comparisons(
            env_aggregated + social_aggregated
        )

        # Step 5: Evaluate Theory of Change
        toc = theory_of_change
        if toc is not None:
            toc = self._evaluate_toc_completeness(toc)
            self._toc = toc

        # Step 6: Assess additionality
        additionality = None
        if additionality_inputs:
            additionality = self._assess_additionality(additionality_inputs)

        # Step 7: Data quality metrics
        total_kpis = len(env_aggregated) + len(social_aggregated)
        kpis_with_data = sum(
            1 for k in env_aggregated + social_aggregated
            if k.value != 0.0
        )
        reported_count = sum(
            1 for k in kpis if k.data_quality == "reported"
        )
        data_coverage = _safe_pct(kpis_with_data, total_kpis)
        reported_pct = _safe_pct(reported_count, len(kpis))

        # Step 8: Improvement counts
        improved = sum(1 for c in comparisons if c.on_track)
        deteriorated = sum(
            1 for c in comparisons
            if not c.on_track and c.direction != "unchanged"
        )

        # Step 9: Unique holding count
        holding_ids = set(k.holding_id for k in kpis if k.holding_id)

        processing_ms = (utcnow() - start).total_seconds() * 1000.0

        result = ImpactResult(
            product_name=self.config.product_name,
            environmental_kpis=env_aggregated,
            social_kpis=social_aggregated,
            total_kpis_tracked=total_kpis,
            env_kpi_count=len(env_aggregated),
            social_kpi_count=len(social_aggregated),
            sdg_contributions=sdg_contributions,
            primary_sdgs=primary_sdgs,
            sdg_coverage_count=sum(
                1 for s in sdg_contributions
                if s.contribution_score > 0
            ),
            theory_of_change=toc,
            additionality=additionality,
            period_comparisons=comparisons,
            kpis_improved_count=improved,
            kpis_deteriorated_count=deteriorated,
            data_coverage_pct=_round_val(data_coverage, 4),
            reported_data_pct=_round_val(reported_pct, 4),
            total_holdings=len(holding_ids),
            processing_time_ms=processing_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Impact assessed: env_kpis=%d, social_kpis=%d, "
            "sdg_coverage=%d, primary_sdgs=%d, improved=%d in %.0fms",
            len(env_aggregated),
            len(social_aggregated),
            result.sdg_coverage_count,
            len(primary_sdgs),
            improved,
            processing_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Individual operations
    # ------------------------------------------------------------------

    def aggregate_kpi(
        self,
        kpis: List[ImpactKPI],
        kpi_id: str,
    ) -> ImpactKPI:
        """Aggregate a single KPI across all holdings.

        Sums attributed values across holdings for the specified KPI.

        Args:
            kpis: List of holding-level KPI data.
            kpi_id: The KPI identifier to aggregate.

        Returns:
            Portfolio-level aggregated ImpactKPI.
        """
        filtered = [k for k in kpis if k.kpi_id == kpi_id]
        if not filtered:
            return ImpactKPI(kpi_id=kpi_id, value=0.0)

        total_value = sum(k.attributed_value for k in filtered)
        total_raw = sum(k.value for k in filtered)

        # Use first entry for metadata
        first = filtered[0]

        # Compute prior value if available
        prior_sum: Optional[float] = None
        if any(k.prior_value is not None for k in filtered):
            prior_sum = sum(
                k.prior_value * k.attribution_factor
                for k in filtered
                if k.prior_value is not None
            )

        return ImpactKPI(
            kpi_id=kpi_id,
            value=_round_val(total_value, 4),
            prior_value=(
                _round_val(prior_sum, 4)
                if prior_sum is not None
                else None
            ),
            unit=first.unit,
            category=first.category,
            data_quality=self._dominant_quality(filtered),
            attribution_factor=1.0,
            attributed_value=_round_val(total_value, 4),
            reporting_year=first.reporting_year,
            source="portfolio_aggregation",
        )

    def compute_sdg_score(
        self,
        portfolio_kpis: List[ImpactKPI],
        sdg: SDGGoal,
    ) -> float:
        """Compute contribution score for a specific SDG.

        Aggregates all KPIs linked to the given SDG and computes
        a normalized score (0-100).

        Args:
            portfolio_kpis: Portfolio-level aggregated KPIs.
            sdg: The SDG goal to score.

        Returns:
            Contribution score (0-100).
        """
        linked_kpi_ids = []
        for kpi_id, defn in ALL_KPI_DEFINITIONS.items():
            if sdg in defn.get("sdgs", []):
                linked_kpi_ids.append(kpi_id)

        if not linked_kpi_ids:
            return 0.0

        matched_kpis = [
            k for k in portfolio_kpis if k.kpi_id in linked_kpi_ids
        ]
        if not matched_kpis:
            return 0.0

        # Score: proportion of linked KPIs with positive attributed value
        positive_count = sum(
            1 for k in matched_kpis if k.attributed_value > 0
        )
        base_score = _safe_pct(positive_count, len(linked_kpi_ids))

        # Bonus for YoY improvement
        improvement_bonus = 0.0
        for k in matched_kpis:
            if k.prior_value is not None and k.prior_value > 0:
                change = _yoy_change(k.attributed_value, k.prior_value)
                if change > 0:
                    improvement_bonus += min(change / 10.0, 5.0)

        return _round_val(min(100.0, base_score + improvement_bonus), 4)

    def update_kpi(
        self,
        update: KPIUpdate,
        existing_kpis: List[ImpactKPI],
    ) -> List[ImpactKPI]:
        """Apply a KPI update to the existing KPI list.

        Finds the matching KPI and updates its value, preserving the
        old value as prior_value for comparison.

        Args:
            update: The KPI update payload.
            existing_kpis: Current list of KPIs.

        Returns:
            Updated KPI list with the change applied.
        """
        updated = []
        found = False

        for kpi in existing_kpis:
            if (kpi.kpi_id == update.kpi_id
                    and kpi.holding_id == update.holding_id):
                new_kpi = ImpactKPI(
                    kpi_id=kpi.kpi_id,
                    holding_id=kpi.holding_id,
                    value=update.new_value,
                    prior_value=kpi.value,
                    unit=kpi.unit,
                    category=kpi.category,
                    data_quality=update.data_quality,
                    attribution_factor=kpi.attribution_factor,
                    reporting_year=kpi.reporting_year,
                    source=update.source,
                )
                updated.append(new_kpi)
                found = True
            else:
                updated.append(kpi)

        if not found:
            logger.warning(
                "KPI update target not found: kpi_id=%s, holding_id=%s",
                update.kpi_id,
                update.holding_id,
            )

        return updated

    def get_kpi_definitions(self) -> List[KPIDefinition]:
        """Return all KPI definitions from the registry.

        Returns:
            List of KPIDefinition with full metadata.
        """
        definitions: List[KPIDefinition] = []
        for kpi_id, defn in ALL_KPI_DEFINITIONS.items():
            definitions.append(KPIDefinition(
                kpi_id=kpi_id,
                name=defn["name"],
                unit=defn["unit"],
                category=defn["category"],
                sdgs=defn.get("sdgs", []),
                higher_is_better=defn.get("higher_is_better", True),
            ))
        return definitions

    # ------------------------------------------------------------------
    # Internal: KPI aggregation
    # ------------------------------------------------------------------

    def _aggregate_kpis(
        self,
        kpis: List[ImpactKPI],
        category: ImpactCategory,
    ) -> List[ImpactKPI]:
        """Aggregate KPIs to portfolio level for a given category.

        Groups by kpi_id and sums attributed values.

        Args:
            kpis: List of holding-level KPIs.
            category: Category to filter by.

        Returns:
            List of portfolio-level aggregated KPIs.
        """
        # Determine which KPI IDs to track
        if category == ImpactCategory.ENVIRONMENTAL:
            tracked_ids = self.config.tracked_env_kpis
        else:
            tracked_ids = self.config.tracked_social_kpis

        # Filter and group
        category_kpis = [
            k for k in kpis
            if k.kpi_id in tracked_ids or k.category == category
        ]

        # Group by kpi_id
        groups: Dict[str, List[ImpactKPI]] = defaultdict(list)
        for k in category_kpis:
            groups[k.kpi_id].append(k)

        # Aggregate each group
        aggregated: List[ImpactKPI] = []
        for kpi_id in tracked_ids:
            if kpi_id in groups:
                agg = self.aggregate_kpi(groups[kpi_id], kpi_id)
                # Ensure category is set from definition
                defn = ALL_KPI_DEFINITIONS.get(kpi_id, {})
                agg.category = defn.get("category", category)
                if not agg.unit and "unit" in defn:
                    agg.unit = defn["unit"]
                aggregated.append(agg)
            else:
                # No data for this KPI
                defn = ALL_KPI_DEFINITIONS.get(kpi_id, {})
                aggregated.append(ImpactKPI(
                    kpi_id=kpi_id,
                    value=0.0,
                    unit=defn.get("unit", ""),
                    category=defn.get("category", category),
                    data_quality="not_available",
                    source="no_data",
                ))

        return aggregated

    # ------------------------------------------------------------------
    # Internal: SDG contribution computation
    # ------------------------------------------------------------------

    def _compute_sdg_contributions(
        self,
        portfolio_kpis: List[ImpactKPI],
    ) -> List[SDGContribution]:
        """Compute contribution scores for all 17 SDGs.

        Maps portfolio KPIs to SDGs using the static registry and
        computes normalized scores.

        Args:
            portfolio_kpis: Portfolio-level aggregated KPIs.

        Returns:
            List of SDGContribution, one per SDG goal.
        """
        contributions: List[SDGContribution] = []

        for sdg in SDGGoal:
            # Find KPIs linked to this SDG
            linked_kpi_ids: List[str] = []
            for kpi_id, defn in ALL_KPI_DEFINITIONS.items():
                if sdg in defn.get("sdgs", []):
                    linked_kpi_ids.append(kpi_id)

            # Compute contribution score
            score = self.compute_sdg_score(portfolio_kpis, sdg)

            # Sum attributed values for linked KPIs
            total_attributed = sum(
                k.attributed_value
                for k in portfolio_kpis
                if k.kpi_id in linked_kpi_ids
            )

            # Coverage
            matched_count = sum(
                1 for k in portfolio_kpis
                if k.kpi_id in linked_kpi_ids and k.value != 0.0
            )
            coverage = _safe_pct(
                matched_count, len(linked_kpi_ids)
            ) if linked_kpi_ids else 0.0

            # YoY change
            yoy = 0.0
            prior_total = 0.0
            current_total = 0.0
            for k in portfolio_kpis:
                if k.kpi_id in linked_kpi_ids:
                    current_total += k.attributed_value
                    if k.prior_value is not None:
                        prior_total += k.prior_value
            if prior_total > 0:
                yoy = _yoy_change(current_total, prior_total)

            contrib = SDGContribution(
                sdg=sdg,
                sdg_name=_SDG_NAMES.get(sdg, sdg.value),
                sdg_number=_SDG_NUMBERS.get(sdg, 0),
                contributing_kpis=linked_kpi_ids,
                kpi_count=len(linked_kpi_ids),
                contribution_score=_round_val(score, 4),
                total_attributed_value=_round_val(total_attributed, 4),
                coverage_pct=_round_val(coverage, 4),
                yoy_change_pct=_round_val(yoy, 4),
            )
            contrib.provenance_hash = _compute_hash(contrib)
            contributions.append(contrib)

        return contributions

    # ------------------------------------------------------------------
    # Internal: Period comparisons
    # ------------------------------------------------------------------

    def _compute_period_comparisons(
        self,
        portfolio_kpis: List[ImpactKPI],
    ) -> List[PeriodComparison]:
        """Compute year-on-year comparisons for all KPIs with prior data.

        Args:
            portfolio_kpis: Portfolio-level aggregated KPIs.

        Returns:
            List of PeriodComparison for KPIs with prior values.
        """
        comparisons: List[PeriodComparison] = []

        for kpi in portfolio_kpis:
            if kpi.prior_value is None:
                continue

            current = kpi.attributed_value
            prior = kpi.prior_value
            absolute_change = current - prior
            pct_change = _yoy_change(current, prior)

            # Determine direction
            defn = ALL_KPI_DEFINITIONS.get(kpi.kpi_id, {})
            higher_is_better = defn.get("higher_is_better", True)

            if abs(absolute_change) < 1e-10:
                direction = "unchanged"
                on_track = True
            elif absolute_change > 0:
                direction = "improved" if higher_is_better else "deteriorated"
                on_track = higher_is_better
            else:
                direction = "deteriorated" if higher_is_better else "improved"
                on_track = not higher_is_better

            comp = PeriodComparison(
                kpi_id=kpi.kpi_id,
                kpi_name=defn.get("name", kpi.kpi_id),
                current_period=self.config.current_period_label,
                prior_period=self.config.prior_period_label,
                current_value=_round_val(current, 4),
                prior_value=_round_val(prior, 4),
                absolute_change=_round_val(absolute_change, 4),
                pct_change=_round_val(pct_change, 4),
                direction=direction,
                higher_is_better=higher_is_better,
                on_track=on_track,
            )
            comp.provenance_hash = _compute_hash(comp)
            comparisons.append(comp)

        return comparisons

    # ------------------------------------------------------------------
    # Internal: Theory of Change evaluation
    # ------------------------------------------------------------------

    def _evaluate_toc_completeness(
        self,
        toc: TheoryOfChange,
    ) -> TheoryOfChange:
        """Evaluate Theory of Change completeness score.

        Scores based on: stages populated, assumptions documented,
        evidence sources listed, KPIs linked, and SDGs mapped.

        Args:
            toc: Theory of Change model to evaluate.

        Returns:
            Updated TheoryOfChange with completeness score.
        """
        total_checks = 5
        passed = 0

        # Check 1: All five stages have at least one element
        stages_populated = sum(
            1 for stage_items in toc.stages.values()
            if len(stage_items) > 0
        )
        if stages_populated >= 5:
            passed += 1
        elif stages_populated >= 3:
            passed += 0.5

        # Check 2: Assumptions documented
        if len(toc.assumptions) >= 2:
            passed += 1
        elif len(toc.assumptions) >= 1:
            passed += 0.5

        # Check 3: Evidence sources
        if len(toc.evidence_sources) >= 2:
            passed += 1
        elif len(toc.evidence_sources) >= 1:
            passed += 0.5

        # Check 4: KPIs linked
        linked_kpi_count = sum(
            len(kpis) for kpis in toc.linked_kpis.values()
        )
        if linked_kpi_count >= 3:
            passed += 1
        elif linked_kpi_count >= 1:
            passed += 0.5

        # Check 5: SDGs mapped
        if len(toc.linked_sdgs) >= 2:
            passed += 1
        elif len(toc.linked_sdgs) >= 1:
            passed += 0.5

        toc.completeness_score = _round_val(
            _safe_pct(passed, total_checks), 4
        )
        toc.provenance_hash = _compute_hash(toc)

        logger.info(
            "ToC completeness evaluated: %.1f%% (stages=%d, "
            "assumptions=%d, evidence=%d, kpis=%d, sdgs=%d)",
            toc.completeness_score,
            stages_populated,
            len(toc.assumptions),
            len(toc.evidence_sources),
            linked_kpi_count,
            len(toc.linked_sdgs),
        )
        return toc

    # ------------------------------------------------------------------
    # Internal: Additionality assessment
    # ------------------------------------------------------------------

    def _assess_additionality(
        self,
        inputs: Dict[str, float],
    ) -> AdditionalityResult:
        """Assess investment additionality.

        Uses the IMP five dimensions approach: intentionality,
        contribution, counterfactual, and materiality.

        Args:
            inputs: Dict with keys: intentionality, contribution,
                counterfactual, materiality (each 0-100).

        Returns:
            AdditionalityResult with overall score.
        """
        intentionality = min(100.0, max(0.0, inputs.get("intentionality", 0.0)))
        contribution = min(100.0, max(0.0, inputs.get("contribution", 0.0)))
        counterfactual = min(100.0, max(0.0, inputs.get("counterfactual", 0.0)))
        materiality = min(100.0, max(0.0, inputs.get("materiality", 0.0)))

        # Weighted average (equal weights)
        overall = (
            intentionality * 0.25
            + contribution * 0.25
            + counterfactual * 0.30
            + materiality * 0.20
        )

        result = AdditionalityResult(
            product_name=self.config.product_name,
            intentionality_score=_round_val(intentionality, 4),
            contribution_score=_round_val(contribution, 4),
            counterfactual_score=_round_val(counterfactual, 4),
            materiality_score=_round_val(materiality, 4),
            overall_additionality_score=_round_val(overall, 4),
            assessment_methodology=self.config.additionality_methodology,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Additionality assessed: overall=%.1f%% (intent=%.0f, "
            "contrib=%.0f, counter=%.0f, material=%.0f)",
            overall, intentionality, contribution,
            counterfactual, materiality,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: Data quality helpers
    # ------------------------------------------------------------------

    def _dominant_quality(self, kpis: List[ImpactKPI]) -> str:
        """Determine the dominant data quality level among KPIs.

        Returns the most conservative quality level (worst quality)
        present in the list.

        Args:
            kpis: List of KPIs to assess.

        Returns:
            Dominant quality level string.
        """
        quality_order = ["not_available", "modeled", "estimated", "reported"]
        qualities = [k.data_quality for k in kpis]
        for q in quality_order:
            if q in qualities:
                return q
        return "reported"
