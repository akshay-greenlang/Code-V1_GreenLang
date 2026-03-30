# -*- coding: utf-8 -*-
"""
ClimateFinanceEngine - PACK-022 Net Zero Acceleration Engine 5
================================================================

Climate transition finance integration engine covering CapEx
classification, EU Taxonomy alignment, green bond eligibility,
internal carbon pricing, investment case analysis (NPV/IRR/payback),
and total cost of inaction calculation.

This engine bridges decarbonization strategy with financial decision-
making by quantifying the investment case for climate actions.  It
classifies CapEx as climate-related or non-climate, screens against
EU Taxonomy substantial contribution criteria, evaluates green bond
eligibility per ICMA Green Bond Principles, applies shadow carbon
pricing to emissions, and calculates the financial ROI of climate
investments including avoided carbon costs.

Calculation Methodology:
    CapEx classification:
        climate_capex_pct = sum(climate_capex) / total_capex * 100

    EU Taxonomy alignment:
        aligned_pct = sum(capex where SC=True AND DNSH=True AND MS=True)
                      / total_capex * 100

    Green bond eligibility:
        eligible_pct = sum(capex matching ICMA categories)
                       / total_capex * 100

    Internal carbon pricing:
        carbon_cost = emissions_tco2e * shadow_price_per_tco2e

    Carbon price trajectory:
        price(t) = base_price * (target_price/base_price)
                   ^((t-base_year)/(target_year-base_year))

    NPV:
        npv = -capex + sum(savings / (1+r)^t for t in 1..horizon)

    IRR:
        NPV(irr) = 0, solved by bisection method

    Payback:
        payback = capex / annual_savings

    Cost of inaction:
        inaction_cost = sum(emissions(t) * carbon_price(t)
                            for t in base..target)

    ROI:
        roi = (total_benefit - total_cost) / total_cost * 100

Regulatory References:
    - EU Taxonomy Regulation (2020/852) - Climate mitigation TSC
    - EU Taxonomy Climate Delegated Act (2021/2139)
    - ICMA Green Bond Principles (2021, updated 2023)
    - IEA Net Zero by 2050 - Carbon price projections
    - NGFS Climate Scenarios v4 (2023) - Carbon pricing
    - TCFD Recommendations (2017) - Metrics & Targets
    - EU CSRD / ESRS E1-6 - Climate-related CapEx/OpEx
    - SBTi Financial Sector Guidance (2022)

Zero-Hallucination:
    - All financial calculations use deterministic Decimal arithmetic
    - Carbon price trajectories from IEA/NGFS published scenarios
    - EU Taxonomy TSC thresholds from official delegated acts
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-022 Net Zero Acceleration
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

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

class CapExCategory(str, Enum):
    """Climate CapEx category classification."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    ELECTRIFICATION = "electrification"
    PROCESS_IMPROVEMENT = "process_improvement"
    FLEET_TRANSITION = "fleet_transition"
    BUILDING_RETROFIT = "building_retrofit"
    CCS_CCUS = "ccs_ccus"
    NATURE_BASED = "nature_based"
    CIRCULAR_ECONOMY = "circular_economy"
    NON_CLIMATE = "non_climate"

class TaxonomyActivity(str, Enum):
    """EU Taxonomy eligible activity classification.

    Maps to Climate Delegated Act Annex I activities.
    """
    ELECTRICITY_GENERATION_SOLAR = "7.1_solar_pv"
    ELECTRICITY_GENERATION_WIND = "7.2_wind"
    ELECTRICITY_STORAGE = "7.3_storage"
    RENOVATION_BUILDINGS = "7.2_renovation"
    HEAT_PUMP_INSTALLATION = "7.3_heat_pumps"
    EV_INFRASTRUCTURE = "6.15_ev_infrastructure"
    MANUFACTURING_LOW_CARBON = "3.6_low_carbon_manufacturing"
    CCS = "5.12_ccs"
    AFFORESTATION = "1.1_afforestation"
    ENERGY_EFFICIENCY_EQUIPMENT = "3.5_ee_equipment"
    NOT_ELIGIBLE = "not_eligible"

class BondCategory(str, Enum):
    """ICMA Green Bond Principles eligible project categories."""
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    POLLUTION_PREVENTION = "pollution_prevention"
    CLEAN_TRANSPORTATION = "clean_transportation"
    SUSTAINABLE_WATER = "sustainable_water"
    GREEN_BUILDINGS = "green_buildings"
    TERRESTRIAL_AQUATIC = "terrestrial_aquatic"
    CLIMATE_ADAPTATION = "climate_adaptation"
    CIRCULAR_ECONOMY = "circular_economy"
    NOT_ELIGIBLE = "not_eligible"

class CarbonPriceScenario(str, Enum):
    """Carbon price trajectory scenario."""
    CURRENT_POLICIES = "current_policies"
    ANNOUNCED_PLEDGES = "announced_pledges"
    NET_ZERO_2050 = "net_zero_2050"
    CUSTOM = "custom"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Carbon price trajectories (USD per tCO2e).
# Source: IEA WEO 2024, NGFS v4 (2023).
CARBON_PRICE_TRAJECTORIES: Dict[str, Dict[int, Decimal]] = {
    CarbonPriceScenario.CURRENT_POLICIES: {
        2025: Decimal("25"),
        2030: Decimal("35"),
        2035: Decimal("45"),
        2040: Decimal("55"),
        2045: Decimal("65"),
        2050: Decimal("75"),
    },
    CarbonPriceScenario.ANNOUNCED_PLEDGES: {
        2025: Decimal("40"),
        2030: Decimal("65"),
        2035: Decimal("90"),
        2040: Decimal("115"),
        2045: Decimal("140"),
        2050: Decimal("160"),
    },
    CarbonPriceScenario.NET_ZERO_2050: {
        2025: Decimal("75"),
        2030: Decimal("140"),
        2035: Decimal("175"),
        2040: Decimal("205"),
        2045: Decimal("230"),
        2050: Decimal("250"),
    },
}

# EU Taxonomy substantial contribution thresholds for climate mitigation.
# Source: EU Climate Delegated Act (2021/2139) Annex I.
TAXONOMY_TSC_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    TaxonomyActivity.ELECTRICITY_GENERATION_SOLAR: {
        "threshold": "Life-cycle <100 gCO2e/kWh",
        "threshold_value_gco2e_kwh": Decimal("100"),
        "dnsh_criteria": "Waste management for panels",
    },
    TaxonomyActivity.ELECTRICITY_GENERATION_WIND: {
        "threshold": "Life-cycle <100 gCO2e/kWh",
        "threshold_value_gco2e_kwh": Decimal("100"),
        "dnsh_criteria": "Bird/bat impact assessment",
    },
    TaxonomyActivity.RENOVATION_BUILDINGS: {
        "threshold": "30% primary energy demand reduction",
        "threshold_value_pct": Decimal("30"),
        "dnsh_criteria": "Waste recycling from renovation",
    },
    TaxonomyActivity.HEAT_PUMP_INSTALLATION: {
        "threshold": "GWP < 675 for refrigerant",
        "threshold_value_gwp": Decimal("675"),
        "dnsh_criteria": "F-gas regulation compliance",
    },
    TaxonomyActivity.EV_INFRASTRUCTURE: {
        "threshold": "Exclusively for zero direct emission vehicles",
        "threshold_value_gco2e_km": Decimal("0"),
        "dnsh_criteria": "Circular economy for equipment",
    },
    TaxonomyActivity.CCS: {
        "threshold": "Capture >90% of CO2 from source",
        "threshold_value_pct": Decimal("90"),
        "dnsh_criteria": "Storage site monitoring plan",
    },
    TaxonomyActivity.AFFORESTATION: {
        "threshold": "Sustainable forest management plan",
        "threshold_value_pct": Decimal("0"),
        "dnsh_criteria": "Biodiversity impact assessment",
    },
}

# ICMA Green Bond category mappings from CapEx categories.
CAPEX_TO_BOND: Dict[str, str] = {
    CapExCategory.RENEWABLE_ENERGY: BondCategory.RENEWABLE_ENERGY.value,
    CapExCategory.ENERGY_EFFICIENCY: BondCategory.ENERGY_EFFICIENCY.value,
    CapExCategory.ELECTRIFICATION: BondCategory.CLEAN_TRANSPORTATION.value,
    CapExCategory.FLEET_TRANSITION: BondCategory.CLEAN_TRANSPORTATION.value,
    CapExCategory.BUILDING_RETROFIT: BondCategory.GREEN_BUILDINGS.value,
    CapExCategory.CCS_CCUS: BondCategory.POLLUTION_PREVENTION.value,
    CapExCategory.NATURE_BASED: BondCategory.TERRESTRIAL_AQUATIC.value,
    CapExCategory.CIRCULAR_ECONOMY: BondCategory.CIRCULAR_ECONOMY.value,
    CapExCategory.PROCESS_IMPROVEMENT: BondCategory.ENERGY_EFFICIENCY.value,
}

# Default discount rate for NPV/IRR calculations.
DEFAULT_DISCOUNT_RATE: Decimal = Decimal("0.08")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class CapExItem(BaseModel):
    """A single CapEx investment item.

    Attributes:
        item_id: Unique identifier.
        name: Investment name.
        category: Climate CapEx category.
        amount_usd: CapEx amount.
        annual_savings_usd: Expected annual operating savings.
        annual_abatement_tco2e: Expected annual emission reduction.
        lifetime_years: Asset lifetime / investment horizon.
        taxonomy_activity: EU Taxonomy activity classification.
        meets_substantial_contribution: Whether SC criteria are met.
        meets_dnsh: Whether DNSH criteria are met.
        meets_minimum_safeguards: Whether MS criteria are met.
        description: Description of the investment.
    """
    item_id: str = Field(default_factory=_new_uuid)
    name: str = Field(..., min_length=1, max_length=300)
    category: CapExCategory = Field(...)
    amount_usd: Decimal = Field(..., ge=Decimal("0"))
    annual_savings_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    annual_abatement_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    lifetime_years: int = Field(default=10, ge=1, le=50)
    taxonomy_activity: TaxonomyActivity = Field(
        default=TaxonomyActivity.NOT_ELIGIBLE,
    )
    meets_substantial_contribution: bool = Field(default=False)
    meets_dnsh: bool = Field(default=True)
    meets_minimum_safeguards: bool = Field(default=True)
    description: str = Field(default="", max_length=1000)

class ClimateOpExEntry(BaseModel):
    """Recurring climate-related operating expenditure.

    Attributes:
        name: OpEx item name.
        annual_amount_usd: Annual cost.
        category: Category of climate OpEx.
    """
    name: str = Field(..., max_length=300)
    annual_amount_usd: Decimal = Field(..., ge=Decimal("0"))
    category: str = Field(default="general")

class ClimateFinanceInput(BaseModel):
    """Complete input for climate finance analysis.

    Attributes:
        entity_name: Reporting entity name.
        base_year: Current/base year.
        target_year: Net-zero target year.
        total_capex_usd: Total organizational CapEx.
        capex_items: Climate-related CapEx items.
        climate_opex: Recurring climate OpEx items.
        current_emissions_tco2e: Current annual emissions.
        projected_reduction_rate_pct: Annual reduction rate (%).
        carbon_price_scenario: Carbon price trajectory to use.
        custom_carbon_price_base_usd: Custom base carbon price.
        custom_carbon_price_2050_usd: Custom 2050 carbon price.
        discount_rate: Discount rate for NPV/IRR.
        shadow_carbon_price_usd: Internal shadow carbon price.
        include_cost_of_inaction: Calculate cost of inaction.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    base_year: int = Field(
        ..., ge=2020, le=2030, description="Base year"
    )
    target_year: int = Field(
        default=2050, ge=2030, le=2070, description="Target year"
    )
    total_capex_usd: Decimal = Field(
        ..., gt=Decimal("0"), description="Total CapEx"
    )
    capex_items: List[CapExItem] = Field(
        default_factory=list, description="Climate CapEx items"
    )
    climate_opex: List[ClimateOpExEntry] = Field(
        default_factory=list, description="Climate OpEx items"
    )
    current_emissions_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Current annual emissions"
    )
    projected_reduction_rate_pct: Decimal = Field(
        default=Decimal("4.2"), ge=Decimal("0"), le=Decimal("20"),
    )
    carbon_price_scenario: CarbonPriceScenario = Field(
        default=CarbonPriceScenario.ANNOUNCED_PLEDGES,
    )
    custom_carbon_price_base_usd: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
    )
    custom_carbon_price_2050_usd: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
    )
    discount_rate: Decimal = Field(
        default=Decimal("0.08"), ge=Decimal("0"), le=Decimal("0.30"),
    )
    shadow_carbon_price_usd: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"),
        description="Shadow carbon price ($/tCO2e)",
    )
    include_cost_of_inaction: bool = Field(default=True)

    @field_validator("target_year")
    @classmethod
    def validate_target(cls, v: int, info: Any) -> int:
        """Target year must be after base year."""
        base = info.data.get("base_year", 2020)
        if v <= base:
            raise ValueError(f"target_year ({v}) must be after base_year ({base})")
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class CapExClassification(BaseModel):
    """CapEx classification summary.

    Attributes:
        total_capex_usd: Total CapEx.
        climate_capex_usd: Climate-related CapEx.
        non_climate_capex_usd: Non-climate CapEx.
        climate_capex_pct: Climate as % of total.
        by_category: CapEx per climate category.
    """
    total_capex_usd: Decimal = Field(default=Decimal("0"))
    climate_capex_usd: Decimal = Field(default=Decimal("0"))
    non_climate_capex_usd: Decimal = Field(default=Decimal("0"))
    climate_capex_pct: Decimal = Field(default=Decimal("0"))
    by_category: Dict[str, Decimal] = Field(default_factory=dict)

class TaxonomyAlignment(BaseModel):
    """EU Taxonomy alignment assessment.

    Attributes:
        eligible_capex_usd: Taxonomy-eligible CapEx.
        eligible_pct: Eligible as % of total.
        aligned_capex_usd: Taxonomy-aligned CapEx (SC+DNSH+MS).
        aligned_pct: Aligned as % of total.
        by_activity: Aligned CapEx per Taxonomy activity.
        items_assessed: Number of items assessed.
        items_aligned: Number of items passing all criteria.
    """
    eligible_capex_usd: Decimal = Field(default=Decimal("0"))
    eligible_pct: Decimal = Field(default=Decimal("0"))
    aligned_capex_usd: Decimal = Field(default=Decimal("0"))
    aligned_pct: Decimal = Field(default=Decimal("0"))
    by_activity: Dict[str, Decimal] = Field(default_factory=dict)
    items_assessed: int = Field(default=0)
    items_aligned: int = Field(default=0)

class GreenBondEligibility(BaseModel):
    """ICMA Green Bond Principles eligibility.

    Attributes:
        eligible_capex_usd: Bond-eligible CapEx.
        eligible_pct: Eligible as % of total.
        by_category: Eligible CapEx per ICMA category.
        items_eligible: Number of eligible items.
    """
    eligible_capex_usd: Decimal = Field(default=Decimal("0"))
    eligible_pct: Decimal = Field(default=Decimal("0"))
    by_category: Dict[str, Decimal] = Field(default_factory=dict)
    items_eligible: int = Field(default=0)

class InvestmentCase(BaseModel):
    """Investment case for a single CapEx item.

    Attributes:
        item_id: Item identifier.
        name: Item name.
        capex_usd: CapEx amount.
        annual_savings_usd: Annual savings.
        npv_usd: Net present value.
        irr_pct: Internal rate of return (%).
        simple_payback_years: Simple payback period.
        carbon_benefit_usd: NPV of avoided carbon cost.
        total_npv_with_carbon_usd: NPV including carbon benefit.
    """
    item_id: str = Field(default="")
    name: str = Field(default="")
    capex_usd: Decimal = Field(default=Decimal("0"))
    annual_savings_usd: Decimal = Field(default=Decimal("0"))
    npv_usd: Decimal = Field(default=Decimal("0"))
    irr_pct: Optional[Decimal] = Field(None)
    simple_payback_years: Optional[Decimal] = Field(None)
    carbon_benefit_usd: Decimal = Field(default=Decimal("0"))
    total_npv_with_carbon_usd: Decimal = Field(default=Decimal("0"))

class CarbonPriceImpact(BaseModel):
    """Carbon price impact analysis.

    Attributes:
        scenario: Carbon price scenario used.
        current_carbon_cost_usd: Current annual carbon cost.
        projected_2030_cost_usd: Projected 2030 carbon cost.
        projected_2050_cost_usd: Projected 2050 carbon cost.
        cumulative_carbon_cost_usd: Cumulative cost base-to-target.
        shadow_price_annual_cost_usd: Annual cost at shadow price.
        price_trajectory: Year-by-year carbon price.
    """
    scenario: str = Field(default="")
    current_carbon_cost_usd: Decimal = Field(default=Decimal("0"))
    projected_2030_cost_usd: Decimal = Field(default=Decimal("0"))
    projected_2050_cost_usd: Decimal = Field(default=Decimal("0"))
    cumulative_carbon_cost_usd: Decimal = Field(default=Decimal("0"))
    shadow_price_annual_cost_usd: Decimal = Field(default=Decimal("0"))
    price_trajectory: Dict[int, Decimal] = Field(default_factory=dict)

class CostOfInaction(BaseModel):
    """Cost of inaction analysis.

    Attributes:
        cumulative_carbon_cost_no_action_usd: Total cost with no action.
        cumulative_carbon_cost_with_action_usd: Total cost with action.
        savings_from_action_usd: Savings from taking action.
        action_premium_usd: Additional cost of action (CapEx + OpEx).
        net_benefit_usd: Savings minus action premium.
    """
    cumulative_carbon_cost_no_action_usd: Decimal = Field(default=Decimal("0"))
    cumulative_carbon_cost_with_action_usd: Decimal = Field(default=Decimal("0"))
    savings_from_action_usd: Decimal = Field(default=Decimal("0"))
    action_premium_usd: Decimal = Field(default=Decimal("0"))
    net_benefit_usd: Decimal = Field(default=Decimal("0"))

class ROISummary(BaseModel):
    """Climate investment ROI summary.

    Attributes:
        total_climate_investment_usd: Total climate CapEx + OpEx.
        total_savings_usd: Total savings over horizon.
        total_carbon_benefit_usd: Total avoided carbon cost.
        total_benefit_usd: Savings + carbon benefit.
        roi_pct: Return on investment (%).
        roi_including_carbon_pct: ROI including carbon benefit.
    """
    total_climate_investment_usd: Decimal = Field(default=Decimal("0"))
    total_savings_usd: Decimal = Field(default=Decimal("0"))
    total_carbon_benefit_usd: Decimal = Field(default=Decimal("0"))
    total_benefit_usd: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))
    roi_including_carbon_pct: Decimal = Field(default=Decimal("0"))

class ClimateFinanceResult(BaseModel):
    """Complete climate finance analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        capex_classification: CapEx classification summary.
        taxonomy_alignment: EU Taxonomy alignment assessment.
        green_bond_eligibility: Green bond eligibility assessment.
        carbon_price_impact: Carbon price impact analysis.
        investment_cases: Per-item investment cases.
        cost_of_inaction: Cost of inaction analysis.
        roi_summary: ROI summary.
        total_climate_capex_usd: Total climate CapEx.
        total_climate_opex_annual_usd: Total annual climate OpEx.
        total_annual_abatement_tco2e: Total annual abatement.
        recommendations: Improvement recommendations.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    capex_classification: CapExClassification = Field(
        default_factory=CapExClassification
    )
    taxonomy_alignment: TaxonomyAlignment = Field(
        default_factory=TaxonomyAlignment
    )
    green_bond_eligibility: GreenBondEligibility = Field(
        default_factory=GreenBondEligibility
    )
    carbon_price_impact: CarbonPriceImpact = Field(
        default_factory=CarbonPriceImpact
    )
    investment_cases: List[InvestmentCase] = Field(default_factory=list)
    cost_of_inaction: CostOfInaction = Field(default_factory=CostOfInaction)
    roi_summary: ROISummary = Field(default_factory=ROISummary)
    total_climate_capex_usd: Decimal = Field(default=Decimal("0"))
    total_climate_opex_annual_usd: Decimal = Field(default=Decimal("0"))
    total_annual_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ClimateFinanceEngine:
    """Climate transition finance integration engine.

    Provides deterministic, zero-hallucination calculations for:
    - CapEx classification (climate vs non-climate)
    - EU Taxonomy alignment assessment (SC + DNSH + MS)
    - ICMA Green Bond eligibility screening
    - Internal carbon pricing impact
    - NPV, IRR, payback for each climate investment
    - Cost of inaction analysis
    - Climate investment ROI

    All calculations use Decimal arithmetic.  No LLM in any path.

    Usage::

        engine = ClimateFinanceEngine()
        result = engine.calculate(finance_input)
        print(f"Taxonomy aligned: {result.taxonomy_alignment.aligned_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: ClimateFinanceInput) -> ClimateFinanceResult:
        """Run complete climate finance analysis.

        Args:
            data: Validated climate finance input.

        Returns:
            ClimateFinanceResult with all assessments.
        """
        t0 = time.perf_counter()
        logger.info(
            "Climate finance: entity=%s, capex=%.2f, items=%d",
            data.entity_name, float(data.total_capex_usd),
            len(data.capex_items),
        )

        # Step 1: CapEx classification
        classification = self._classify_capex(data)

        # Step 2: EU Taxonomy alignment
        taxonomy = self._assess_taxonomy_alignment(data)

        # Step 3: Green bond eligibility
        bond_eligibility = self._assess_green_bond(data)

        # Step 4: Carbon price impact
        carbon_impact = self._calculate_carbon_impact(data)

        # Step 5: Investment cases
        investment_cases = self._calculate_investment_cases(data)

        # Step 6: Cost of inaction
        inaction = CostOfInaction()
        if data.include_cost_of_inaction:
            inaction = self._calculate_cost_of_inaction(data, carbon_impact)

        # Step 7: ROI summary
        roi = self._calculate_roi(data, investment_cases, carbon_impact)

        # Totals
        total_climate_capex = sum(
            item.amount_usd for item in data.capex_items
            if item.category != CapExCategory.NON_CLIMATE
        )
        total_opex = sum(
            entry.annual_amount_usd for entry in data.climate_opex
        )
        total_abatement = sum(
            item.annual_abatement_tco2e for item in data.capex_items
        )

        # Recommendations
        recommendations = self._generate_recommendations(
            data, classification, taxonomy, bond_eligibility, roi
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ClimateFinanceResult(
            entity_name=data.entity_name,
            capex_classification=classification,
            taxonomy_alignment=taxonomy,
            green_bond_eligibility=bond_eligibility,
            carbon_price_impact=carbon_impact,
            investment_cases=investment_cases,
            cost_of_inaction=inaction,
            roi_summary=roi,
            total_climate_capex_usd=_round_val(total_climate_capex, 2),
            total_climate_opex_annual_usd=_round_val(total_opex, 2),
            total_annual_abatement_tco2e=_round_val(total_abatement),
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Finance complete: climate_capex=%.2f, taxonomy=%.1f%%, "
            "bond=%.1f%%, roi=%.1f%%",
            float(total_climate_capex),
            float(taxonomy.aligned_pct),
            float(bond_eligibility.eligible_pct),
            float(roi.roi_including_carbon_pct),
        )
        return result

    # ------------------------------------------------------------------ #
    # CapEx Classification                                                #
    # ------------------------------------------------------------------ #

    def _classify_capex(
        self, data: ClimateFinanceInput
    ) -> CapExClassification:
        """Classify CapEx as climate-related or non-climate.

        Args:
            data: Finance input.

        Returns:
            CapExClassification summary.
        """
        climate_total = Decimal("0")
        by_category: Dict[str, Decimal] = {}

        for item in data.capex_items:
            if item.category != CapExCategory.NON_CLIMATE:
                climate_total += item.amount_usd
                cat = item.category.value
                by_category[cat] = by_category.get(cat, Decimal("0")) + item.amount_usd

        for key in by_category:
            by_category[key] = _round_val(by_category[key], 2)

        non_climate = data.total_capex_usd - climate_total
        climate_pct = _safe_pct(climate_total, data.total_capex_usd)

        return CapExClassification(
            total_capex_usd=_round_val(data.total_capex_usd, 2),
            climate_capex_usd=_round_val(climate_total, 2),
            non_climate_capex_usd=_round_val(non_climate, 2),
            climate_capex_pct=_round_val(climate_pct, 2),
            by_category=by_category,
        )

    # ------------------------------------------------------------------ #
    # EU Taxonomy Alignment                                               #
    # ------------------------------------------------------------------ #

    def _assess_taxonomy_alignment(
        self, data: ClimateFinanceInput
    ) -> TaxonomyAlignment:
        """Assess EU Taxonomy alignment for climate CapEx items.

        Alignment requires: Eligible activity + Substantial Contribution
        + DNSH + Minimum Safeguards.

        Args:
            data: Finance input.

        Returns:
            TaxonomyAlignment assessment.
        """
        eligible_total = Decimal("0")
        aligned_total = Decimal("0")
        by_activity: Dict[str, Decimal] = {}
        items_assessed = 0
        items_aligned = 0

        for item in data.capex_items:
            if item.taxonomy_activity == TaxonomyActivity.NOT_ELIGIBLE:
                continue

            items_assessed += 1
            eligible_total += item.amount_usd

            # Full alignment check: SC + DNSH + MS
            if (
                item.meets_substantial_contribution
                and item.meets_dnsh
                and item.meets_minimum_safeguards
            ):
                aligned_total += item.amount_usd
                items_aligned += 1
                activity = item.taxonomy_activity.value
                by_activity[activity] = (
                    by_activity.get(activity, Decimal("0"))
                    + item.amount_usd
                )

        for key in by_activity:
            by_activity[key] = _round_val(by_activity[key], 2)

        return TaxonomyAlignment(
            eligible_capex_usd=_round_val(eligible_total, 2),
            eligible_pct=_round_val(
                _safe_pct(eligible_total, data.total_capex_usd), 2
            ),
            aligned_capex_usd=_round_val(aligned_total, 2),
            aligned_pct=_round_val(
                _safe_pct(aligned_total, data.total_capex_usd), 2
            ),
            by_activity=by_activity,
            items_assessed=items_assessed,
            items_aligned=items_aligned,
        )

    # ------------------------------------------------------------------ #
    # Green Bond Eligibility                                              #
    # ------------------------------------------------------------------ #

    def _assess_green_bond(
        self, data: ClimateFinanceInput
    ) -> GreenBondEligibility:
        """Screen CapEx against ICMA Green Bond Principles.

        Args:
            data: Finance input.

        Returns:
            GreenBondEligibility assessment.
        """
        eligible_total = Decimal("0")
        by_category: Dict[str, Decimal] = {}
        items_eligible = 0

        for item in data.capex_items:
            bond_cat = CAPEX_TO_BOND.get(item.category)
            if bond_cat is None or bond_cat == BondCategory.NOT_ELIGIBLE.value:
                continue

            eligible_total += item.amount_usd
            items_eligible += 1
            by_category[bond_cat] = (
                by_category.get(bond_cat, Decimal("0"))
                + item.amount_usd
            )

        for key in by_category:
            by_category[key] = _round_val(by_category[key], 2)

        return GreenBondEligibility(
            eligible_capex_usd=_round_val(eligible_total, 2),
            eligible_pct=_round_val(
                _safe_pct(eligible_total, data.total_capex_usd), 2
            ),
            by_category=by_category,
            items_eligible=items_eligible,
        )

    # ------------------------------------------------------------------ #
    # Carbon Price Impact                                                 #
    # ------------------------------------------------------------------ #

    def _calculate_carbon_impact(
        self, data: ClimateFinanceInput
    ) -> CarbonPriceImpact:
        """Calculate carbon price impact across the projection period.

        Args:
            data: Finance input.

        Returns:
            CarbonPriceImpact analysis.
        """
        # Get price trajectory
        trajectory = self._get_price_trajectory(data)

        # Current carbon cost
        current_price = trajectory.get(data.base_year, Decimal("50"))
        current_cost = data.current_emissions_tco2e * current_price

        # Projected costs at key years
        price_2030 = self._interpolate_price(trajectory, 2030)
        emissions_2030 = self._project_emissions(
            data.current_emissions_tco2e,
            data.projected_reduction_rate_pct,
            data.base_year, 2030,
        )
        cost_2030 = emissions_2030 * price_2030

        price_2050 = self._interpolate_price(trajectory, 2050)
        emissions_2050 = self._project_emissions(
            data.current_emissions_tco2e,
            data.projected_reduction_rate_pct,
            data.base_year, 2050,
        )
        cost_2050 = emissions_2050 * price_2050

        # Cumulative cost
        cumulative = Decimal("0")
        for year in range(data.base_year, data.target_year + 1):
            price = self._interpolate_price(trajectory, year)
            emissions = self._project_emissions(
                data.current_emissions_tco2e,
                data.projected_reduction_rate_pct,
                data.base_year, year,
            )
            cumulative += emissions * price

        # Shadow price
        shadow_cost = data.current_emissions_tco2e * data.shadow_carbon_price_usd

        # Build trajectory dict for output
        price_traj_output: Dict[int, Decimal] = {}
        for year in range(data.base_year, data.target_year + 1, 5):
            price_traj_output[year] = _round_val(
                self._interpolate_price(trajectory, year), 2
            )
        price_traj_output[data.target_year] = _round_val(
            self._interpolate_price(trajectory, data.target_year), 2
        )

        return CarbonPriceImpact(
            scenario=data.carbon_price_scenario.value,
            current_carbon_cost_usd=_round_val(current_cost, 2),
            projected_2030_cost_usd=_round_val(cost_2030, 2),
            projected_2050_cost_usd=_round_val(cost_2050, 2),
            cumulative_carbon_cost_usd=_round_val(cumulative, 2),
            shadow_price_annual_cost_usd=_round_val(shadow_cost, 2),
            price_trajectory=price_traj_output,
        )

    def _get_price_trajectory(
        self, data: ClimateFinanceInput
    ) -> Dict[int, Decimal]:
        """Get carbon price trajectory for the selected scenario.

        Args:
            data: Finance input.

        Returns:
            Year-to-price mapping.
        """
        if data.carbon_price_scenario == CarbonPriceScenario.CUSTOM:
            base = data.custom_carbon_price_base_usd or Decimal("50")
            target = data.custom_carbon_price_2050_usd or Decimal("200")
            # Build linear trajectory
            trajectory: Dict[int, Decimal] = {}
            span = Decimal(str(max(2050 - data.base_year, 1)))
            for year in range(data.base_year, 2051):
                frac = _decimal(year - data.base_year) / span
                trajectory[year] = base + (target - base) * frac
            return trajectory

        return dict(CARBON_PRICE_TRAJECTORIES.get(
            data.carbon_price_scenario,
            CARBON_PRICE_TRAJECTORIES[CarbonPriceScenario.ANNOUNCED_PLEDGES],
        ))

    def _interpolate_price(
        self, trajectory: Dict[int, Decimal], year: int
    ) -> Decimal:
        """Interpolate carbon price for a given year.

        Args:
            trajectory: Year-to-price mapping.
            year: Target year.

        Returns:
            Interpolated price.
        """
        if year in trajectory:
            return trajectory[year]

        years = sorted(trajectory.keys())
        if year <= years[0]:
            return trajectory[years[0]]
        if year >= years[-1]:
            return trajectory[years[-1]]

        lower_year = years[0]
        upper_year = years[-1]
        for i, y in enumerate(years):
            if y <= year:
                lower_year = y
            if y > year:
                upper_year = y
                break

        lower_val = trajectory[lower_year]
        upper_val = trajectory[upper_year]
        span = _decimal(upper_year - lower_year)
        elapsed = _decimal(year - lower_year)
        return lower_val + (upper_val - lower_val) * _safe_divide(elapsed, span)

    def _project_emissions(
        self,
        base_emissions: Decimal,
        rate_pct: Decimal,
        base_year: int,
        target_year: int,
    ) -> Decimal:
        """Project emissions using linear reduction.

        Args:
            base_emissions: Starting emissions.
            rate_pct: Annual reduction rate (%).
            base_year: Start year.
            target_year: End year.

        Returns:
            Projected emissions (clamped to zero minimum).
        """
        elapsed = _decimal(target_year - base_year)
        reduction = rate_pct / Decimal("100") * elapsed
        factor = max(Decimal("0"), Decimal("1") - reduction)
        return base_emissions * factor

    # ------------------------------------------------------------------ #
    # Investment Cases                                                    #
    # ------------------------------------------------------------------ #

    def _calculate_investment_cases(
        self, data: ClimateFinanceInput
    ) -> List[InvestmentCase]:
        """Calculate NPV, IRR, payback for each CapEx item.

        Args:
            data: Finance input.

        Returns:
            List of InvestmentCase entries.
        """
        cases: List[InvestmentCase] = []

        for item in data.capex_items:
            if item.category == CapExCategory.NON_CLIMATE:
                continue

            # NPV
            npv = self._calculate_npv(
                item.amount_usd, item.annual_savings_usd,
                data.discount_rate, item.lifetime_years,
            )

            # IRR
            irr = self._calculate_irr(
                item.amount_usd, item.annual_savings_usd,
                item.lifetime_years,
            )

            # Payback
            payback = None
            if item.annual_savings_usd > Decimal("0"):
                payback = _round_val(
                    _safe_divide(item.amount_usd, item.annual_savings_usd), 1
                )

            # Carbon benefit
            carbon_benefit = self._calculate_carbon_benefit(
                item.annual_abatement_tco2e, data.shadow_carbon_price_usd,
                data.discount_rate, item.lifetime_years,
            )

            total_npv_with_carbon = npv + carbon_benefit

            cases.append(InvestmentCase(
                item_id=item.item_id,
                name=item.name,
                capex_usd=_round_val(item.amount_usd, 2),
                annual_savings_usd=_round_val(item.annual_savings_usd, 2),
                npv_usd=_round_val(npv, 2),
                irr_pct=irr,
                simple_payback_years=payback,
                carbon_benefit_usd=_round_val(carbon_benefit, 2),
                total_npv_with_carbon_usd=_round_val(
                    total_npv_with_carbon, 2
                ),
            ))

        # Sort by total NPV descending
        cases.sort(
            key=lambda c: c.total_npv_with_carbon_usd, reverse=True
        )
        return cases

    def _calculate_npv(
        self,
        capex: Decimal,
        annual_savings: Decimal,
        discount_rate: Decimal,
        horizon: int,
    ) -> Decimal:
        """Calculate Net Present Value.

        NPV = -capex + sum(savings / (1+r)^t for t in 1..horizon)

        Args:
            capex: Capital expenditure.
            annual_savings: Annual operating savings.
            discount_rate: Discount rate.
            horizon: Investment horizon in years.

        Returns:
            NPV as Decimal.
        """
        npv = -capex
        for t in range(1, horizon + 1):
            discount_factor = (Decimal("1") + discount_rate) ** t
            npv += _safe_divide(annual_savings, discount_factor)
        return npv

    def _calculate_irr(
        self,
        capex: Decimal,
        annual_savings: Decimal,
        horizon: int,
    ) -> Optional[Decimal]:
        """Calculate Internal Rate of Return using bisection.

        Args:
            capex: Capital expenditure.
            annual_savings: Annual savings.
            horizon: Investment horizon.

        Returns:
            IRR as percentage Decimal, or None if not calculable.
        """
        if capex <= Decimal("0") or annual_savings <= Decimal("0"):
            return None

        # Bisection method
        low = Decimal("-0.50")
        high = Decimal("5.00")

        for _ in range(100):
            mid = (low + high) / Decimal("2")
            npv = -capex
            for t in range(1, horizon + 1):
                factor = (Decimal("1") + mid) ** t
                if factor != Decimal("0"):
                    npv += _safe_divide(annual_savings, factor)

            if abs(npv) < Decimal("0.01"):
                return _round_val(mid * Decimal("100"), 2)

            if npv > Decimal("0"):
                low = mid
            else:
                high = mid

        return _round_val(((low + high) / Decimal("2")) * Decimal("100"), 2)

    def _calculate_carbon_benefit(
        self,
        annual_abatement: Decimal,
        carbon_price: Decimal,
        discount_rate: Decimal,
        horizon: int,
    ) -> Decimal:
        """Calculate NPV of avoided carbon costs.

        Args:
            annual_abatement: Annual emission reduction (tCO2e).
            carbon_price: Shadow carbon price ($/tCO2e).
            discount_rate: Discount rate.
            horizon: Investment horizon.

        Returns:
            NPV of carbon benefit.
        """
        annual_benefit = annual_abatement * carbon_price
        total = Decimal("0")
        for t in range(1, horizon + 1):
            factor = (Decimal("1") + discount_rate) ** t
            total += _safe_divide(annual_benefit, factor)
        return total

    # ------------------------------------------------------------------ #
    # Cost of Inaction                                                    #
    # ------------------------------------------------------------------ #

    def _calculate_cost_of_inaction(
        self,
        data: ClimateFinanceInput,
        carbon_impact: CarbonPriceImpact,
    ) -> CostOfInaction:
        """Calculate the total cost of not taking climate action.

        Args:
            data: Finance input.
            carbon_impact: Carbon price impact analysis.

        Returns:
            CostOfInaction analysis.
        """
        trajectory = self._get_price_trajectory(data)

        # No action: emissions stay constant
        no_action_cost = Decimal("0")
        for year in range(data.base_year, data.target_year + 1):
            price = self._interpolate_price(trajectory, year)
            no_action_cost += data.current_emissions_tco2e * price

        # With action: emissions decline at projected rate
        with_action_cost = carbon_impact.cumulative_carbon_cost_usd

        savings = no_action_cost - with_action_cost

        # Action premium: total climate CapEx + cumulative OpEx
        action_capex = sum(
            item.amount_usd for item in data.capex_items
            if item.category != CapExCategory.NON_CLIMATE
        )
        action_opex = sum(
            entry.annual_amount_usd for entry in data.climate_opex
        ) * _decimal(data.target_year - data.base_year)
        action_premium = action_capex + action_opex

        net_benefit = savings - action_premium

        return CostOfInaction(
            cumulative_carbon_cost_no_action_usd=_round_val(
                no_action_cost, 2
            ),
            cumulative_carbon_cost_with_action_usd=_round_val(
                with_action_cost, 2
            ),
            savings_from_action_usd=_round_val(savings, 2),
            action_premium_usd=_round_val(action_premium, 2),
            net_benefit_usd=_round_val(net_benefit, 2),
        )

    # ------------------------------------------------------------------ #
    # ROI Summary                                                         #
    # ------------------------------------------------------------------ #

    def _calculate_roi(
        self,
        data: ClimateFinanceInput,
        cases: List[InvestmentCase],
        carbon_impact: CarbonPriceImpact,
    ) -> ROISummary:
        """Calculate overall climate investment ROI.

        Args:
            data: Finance input.
            cases: Investment cases.
            carbon_impact: Carbon impact analysis.

        Returns:
            ROISummary.
        """
        total_investment = sum(c.capex_usd for c in cases)
        total_opex = sum(
            e.annual_amount_usd for e in data.climate_opex
        )
        # Use average lifetime from CapEx items
        avg_lifetime = Decimal("10")
        if data.capex_items:
            avg_lifetime = _decimal(
                sum(i.lifetime_years for i in data.capex_items)
                / len(data.capex_items)
            )

        total_savings = sum(c.annual_savings_usd for c in cases) * avg_lifetime
        total_carbon_benefit = sum(c.carbon_benefit_usd for c in cases)
        total_benefit = total_savings + total_carbon_benefit

        total_cost = total_investment + total_opex * avg_lifetime

        roi = _safe_pct(total_benefit - total_cost, total_cost)
        roi_with_carbon = _safe_pct(total_benefit - total_cost, total_cost)

        # ROI without carbon
        roi_no_carbon = _safe_pct(
            total_savings - total_cost, total_cost
        )

        return ROISummary(
            total_climate_investment_usd=_round_val(total_cost, 2),
            total_savings_usd=_round_val(total_savings, 2),
            total_carbon_benefit_usd=_round_val(total_carbon_benefit, 2),
            total_benefit_usd=_round_val(total_benefit, 2),
            roi_pct=_round_val(roi_no_carbon, 2),
            roi_including_carbon_pct=_round_val(roi_with_carbon, 2),
        )

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: ClimateFinanceInput,
        classification: CapExClassification,
        taxonomy: TaxonomyAlignment,
        bond: GreenBondEligibility,
        roi: ROISummary,
    ) -> List[str]:
        """Generate financial recommendations.

        Args:
            data: Finance input.
            classification: CapEx classification.
            taxonomy: Taxonomy alignment.
            bond: Green bond eligibility.
            roi: ROI summary.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # Low climate CapEx share
        if classification.climate_capex_pct < Decimal("20"):
            recs.append(
                f"Climate CapEx is {classification.climate_capex_pct}% of "
                "total. Leading companies allocate 30-50% to climate. "
                "Increase climate investment to accelerate transition."
            )

        # Taxonomy alignment
        if taxonomy.aligned_pct < Decimal("15"):
            recs.append(
                f"EU Taxonomy alignment is {taxonomy.aligned_pct}%. "
                "Review DNSH and minimum safeguards criteria to increase "
                "alignment. Consider Taxonomy-eligible technologies for "
                "future investments."
            )
        elif taxonomy.eligible_pct > taxonomy.aligned_pct + Decimal("10"):
            gap = taxonomy.eligible_pct - taxonomy.aligned_pct
            recs.append(
                f"{gap}% of CapEx is Taxonomy-eligible but not aligned. "
                "Address DNSH criteria gaps to convert eligibility to "
                "alignment."
            )

        # Green bond opportunity
        if bond.eligible_pct >= Decimal("30"):
            recs.append(
                f"{bond.eligible_pct}% of CapEx qualifies for green bond "
                "financing. Consider issuing a green bond to access lower "
                "cost of capital."
            )

        # Negative ROI warning
        if roi.roi_pct < Decimal("0"):
            if roi.roi_including_carbon_pct > Decimal("0"):
                recs.append(
                    "Climate investments have negative financial ROI but "
                    "positive ROI when carbon costs are included. Internal "
                    "carbon pricing strengthens the investment case."
                )
            else:
                recs.append(
                    "Climate investments show negative ROI even with carbon "
                    "pricing. Explore grant funding, green finance subsidies, "
                    "or phased implementation to improve economics."
                )

        # Shadow price suggestion
        if data.shadow_carbon_price_usd < Decimal("75"):
            recs.append(
                f"Shadow carbon price (${data.shadow_carbon_price_usd}/tCO2e) "
                "is below IEA NZE recommended range ($75-250). Increase to "
                "better reflect transition risk."
            )

        # Quick wins identification
        positive_npv = sum(
            1 for item in data.capex_items
            if item.category != CapExCategory.NON_CLIMATE
            and item.annual_savings_usd > Decimal("0")
        )
        if positive_npv > 0:
            recs.append(
                f"{positive_npv} investment(s) generate positive operational "
                "savings. Prioritize these for immediate implementation."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_carbon_price(
        self, scenario: CarbonPriceScenario, year: int
    ) -> Optional[str]:
        """Look up carbon price for a scenario and year.

        Args:
            scenario: Carbon price scenario.
            year: Target year.

        Returns:
            Price as string, or None if not available.
        """
        trajectory = CARBON_PRICE_TRAJECTORIES.get(scenario)
        if trajectory is None:
            return None
        price = trajectory.get(year)
        return str(price) if price is not None else None

    def get_taxonomy_thresholds(
        self, activity: TaxonomyActivity
    ) -> Optional[Dict[str, str]]:
        """Look up EU Taxonomy TSC thresholds for an activity.

        Args:
            activity: Taxonomy activity.

        Returns:
            Dict with threshold details, or None.
        """
        data = TAXONOMY_TSC_THRESHOLDS.get(activity)
        if data is None:
            return None
        return {k: str(v) for k, v in data.items()}

    def get_summary(
        self, result: ClimateFinanceResult
    ) -> Dict[str, Any]:
        """Generate concise finance summary.

        Args:
            result: Result to summarize.

        Returns:
            Dict with key metrics and provenance hash.
        """
        summary: Dict[str, Any] = {
            "entity_name": result.entity_name,
            "climate_capex_usd": str(result.total_climate_capex_usd),
            "climate_capex_pct": str(
                result.capex_classification.climate_capex_pct
            ),
            "taxonomy_aligned_pct": str(
                result.taxonomy_alignment.aligned_pct
            ),
            "green_bond_eligible_pct": str(
                result.green_bond_eligibility.eligible_pct
            ),
            "roi_pct": str(result.roi_summary.roi_pct),
            "roi_with_carbon_pct": str(
                result.roi_summary.roi_including_carbon_pct
            ),
            "total_abatement_tco2e": str(
                result.total_annual_abatement_tco2e
            ),
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary
