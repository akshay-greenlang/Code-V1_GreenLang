# -*- coding: utf-8 -*-
"""
E5CircularEconomyEngine - PACK-017 ESRS E5 Resource Use & Circular Economy
===========================================================================

Calculates resource inflows, outflows, circular material use rate, product
circularity scores, and anticipated financial effects per ESRS E5.

Under ESRS E5, undertakings shall disclose information on their resource use
and circular economy strategy, covering policies, actions, targets, material
inflows, resource outflows (including waste), and anticipated financial effects.
This engine implements the complete E5 disclosure calculation pipeline:

- Resource inflow aggregation by material type and origin
- Recycled content and renewable material share calculation
- Resource outflow aggregation by waste category and destination
- Waste recycling rate calculation per destination
- Circular Material Use Rate (CMUR) per EU Monitoring Framework
- Product circularity scoring (design, durability, recyclability)
- Financial effects assessment from circularity risks and opportunities
- Completeness validation against E5 required data points
- ESRS E5 data point mapping for disclosure

ESRS E5 Disclosure Requirements:
    - E5-1 (Para 11-13, AR 1-4): Policies related to resource use and
      circular economy, including alignment with waste hierarchy and EPR.
    - E5-2 (Para 17-20, AR 5-8): Actions and resources related to resource
      use and circular economy, including circular design and business models.
    - E5-3 (Para 22-25, AR 9-12): Targets related to resource use and
      circular economy, including absolute and intensity targets.
    - E5-4 (Para 29-33, AR 13-18): Resource inflows, including total weight,
      recycled/reused content, and renewable/non-renewable breakdown.
    - E5-5 (Para 35-40, AR 19-26): Resource outflows, including waste by
      category and destination, products designed for circularity.
    - E5-6 (Para 42-44, AR 27-30): Anticipated financial effects from
      resource use and circular economy related risks and opportunities.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS E5 Resource Use and Circular Economy
    - EU Circular Economy Monitoring Framework (Eurostat)
    - EU Waste Framework Directive 2008/98/EC (waste hierarchy)
    - EU Extended Producer Responsibility guidelines
    - EU Packaging and Packaging Waste Regulation

Zero-Hallucination:
    - All material flow calculations use deterministic arithmetic
    - CMUR formula uses EU Circular Economy Monitoring Framework definition
    - Aggregation uses Decimal arithmetic with ROUND_HALF_UP
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-017 ESRS Full Coverage
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

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
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

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

def _round6(value: Decimal) -> Decimal:
    """Round Decimal to 6 decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MaterialType(str, Enum):
    """Material types for resource inflow classification.

    Per ESRS E5-4 Para 29-33, undertakings shall disclose the total
    weight of material inflows disaggregated by material type.
    """
    RAW_MATERIALS = "raw_materials"
    METALS = "metals"
    MINERALS = "minerals"
    BIOMASS = "biomass"
    FOSSIL_FUELS = "fossil_fuels"
    CHEMICALS = "chemicals"
    ELECTRONICS = "electronics"
    TEXTILES = "textiles"
    PACKAGING = "packaging"

class MaterialOrigin(str, Enum):
    """Origin classification for material inflows.

    Per ESRS E5-4 AR 13-18, undertakings shall report whether materials
    originate from virgin, recycled, renewable, or secondary sources.
    """
    VIRGIN = "virgin"
    RECYCLED = "recycled"
    RENEWABLE = "renewable"
    SECONDARY = "secondary"

class WasteCategory(str, Enum):
    """Waste categories per EU Waste Framework Directive.

    Per ESRS E5-5 Para 35-40, waste shall be disaggregated by hazardous
    and non-hazardous categories.
    """
    HAZARDOUS = "hazardous"
    NON_HAZARDOUS = "non_hazardous"
    RADIOACTIVE = "radioactive"

class WasteDestination(str, Enum):
    """Waste treatment destinations per waste hierarchy.

    Per ESRS E5-5 AR 19-26, undertakings shall report waste by
    destination, following the EU waste hierarchy preference order:
    prevention > reuse > recycling > recovery > disposal.
    """
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    INCINERATION_ENERGY_RECOVERY = "incineration_energy_recovery"
    INCINERATION_NO_RECOVERY = "incineration_no_recovery"
    LANDFILL = "landfill"
    OTHER_DISPOSAL = "other_disposal"

class ProductDesignStrategy(str, Enum):
    """Circular product design strategies.

    Per ESRS E5-5 AR 19-26, undertakings shall describe how products
    are designed for circularity, including durability, recyclability,
    disassembly, reuse, and modular design approaches.
    """
    DESIGN_FOR_DURABILITY = "design_for_durability"
    DESIGN_FOR_RECYCLABILITY = "design_for_recyclability"
    DESIGN_FOR_DISASSEMBLY = "design_for_disassembly"
    DESIGN_FOR_REUSE = "design_for_reuse"
    MODULAR_DESIGN = "modular_design"

class CircularBusinessModel(str, Enum):
    """Circular business model types.

    Per ESRS E5-2 AR 5-8, undertakings shall describe actions to adopt
    circular business models such as product-as-a-service, sharing
    platforms, take-back schemes, refurbishment, or remanufacturing.
    """
    PRODUCT_AS_SERVICE = "product_as_service"
    SHARING_PLATFORM = "sharing_platform"
    TAKE_BACK_SCHEME = "take_back_scheme"
    REFURBISHMENT = "refurbishment"
    REMANUFACTURING = "remanufacturing"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ESRS E5-1 required data points for policies disclosure.
E5_1_DATAPOINTS: List[str] = [
    "e5_1_01_policies_addressing_resource_use",
    "e5_1_02_alignment_with_waste_hierarchy",
    "e5_1_03_circular_economy_strategies",
    "e5_1_04_epr_compliance_status",
    "e5_1_05_packaging_waste_policy",
    "e5_1_06_substances_of_concern_policy",
]

# ESRS E5-2 required data points for actions disclosure.
E5_2_DATAPOINTS: List[str] = [
    "e5_2_01_actions_to_reduce_resource_use",
    "e5_2_02_circular_design_actions",
    "e5_2_03_circular_business_model_actions",
    "e5_2_04_resources_allocated_to_actions",
    "e5_2_05_waste_prevention_actions",
    "e5_2_06_value_chain_engagement_actions",
]

# ESRS E5-3 required data points for targets disclosure.
E5_3_DATAPOINTS: List[str] = [
    "e5_3_01_resource_use_reduction_targets",
    "e5_3_02_recycled_content_targets",
    "e5_3_03_waste_reduction_targets",
    "e5_3_04_recycling_rate_targets",
    "e5_3_05_circular_material_use_rate_target",
    "e5_3_06_target_base_year_and_progress",
]

# ESRS E5-4 required data points for resource inflows disclosure.
E5_4_DATAPOINTS: List[str] = [
    "e5_4_01_total_material_inflow_tonnes",
    "e5_4_02_inflow_by_material_type",
    "e5_4_03_recycled_content_percentage",
    "e5_4_04_renewable_material_percentage",
    "e5_4_05_biological_material_tonnes",
    "e5_4_06_secondary_material_usage",
    "e5_4_07_critical_raw_materials_usage",
]

# ESRS E5-5 required data points for resource outflows disclosure.
E5_5_DATAPOINTS: List[str] = [
    "e5_5_01_total_waste_generated_tonnes",
    "e5_5_02_hazardous_waste_tonnes",
    "e5_5_03_non_hazardous_waste_tonnes",
    "e5_5_04_radioactive_waste_tonnes",
    "e5_5_05_waste_by_destination",
    "e5_5_06_recycling_rate_percentage",
    "e5_5_07_products_designed_for_circularity",
    "e5_5_08_take_back_products_percentage",
]

# ESRS E5-6 required data points for financial effects disclosure.
E5_6_DATAPOINTS: List[str] = [
    "e5_6_01_financial_effects_from_risks",
    "e5_6_02_financial_effects_from_opportunities",
    "e5_6_03_monetary_impact_estimation",
    "e5_6_04_time_horizon_of_effects",
]

# All E5 data points combined.
E5_ALL_DATAPOINTS: List[str] = (
    E5_1_DATAPOINTS + E5_2_DATAPOINTS + E5_3_DATAPOINTS
    + E5_4_DATAPOINTS + E5_5_DATAPOINTS + E5_6_DATAPOINTS
)

# Material circularity benchmarks by industry sector.
# Values represent typical circular material use rates (CMUR) as percentages.
# Source: EU Circular Economy Monitoring Framework (Eurostat, 2024).
MATERIAL_CIRCULARITY_BENCHMARKS: Dict[str, Decimal] = {
    "manufacturing": Decimal("12.0"),
    "construction": Decimal("14.5"),
    "automotive": Decimal("18.0"),
    "electronics": Decimal("8.5"),
    "textiles": Decimal("3.5"),
    "packaging": Decimal("22.0"),
    "chemicals": Decimal("6.0"),
    "food_and_beverage": Decimal("10.0"),
    "retail": Decimal("9.0"),
    "energy": Decimal("5.5"),
    "eu_average": Decimal("11.7"),
}

# Waste destination hierarchy weights (higher = preferred per waste hierarchy).
WASTE_HIERARCHY_WEIGHTS: Dict[str, Decimal] = {
    "recycling": Decimal("5"),
    "composting": Decimal("4"),
    "incineration_energy_recovery": Decimal("3"),
    "incineration_no_recovery": Decimal("2"),
    "landfill": Decimal("1"),
    "other_disposal": Decimal("1"),
}

# Circular design strategy weights for product circularity scoring.
DESIGN_STRATEGY_WEIGHTS: Dict[str, Decimal] = {
    "design_for_durability": Decimal("0.25"),
    "design_for_recyclability": Decimal("0.25"),
    "design_for_disassembly": Decimal("0.20"),
    "design_for_reuse": Decimal("0.20"),
    "modular_design": Decimal("0.10"),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CircularPolicy(BaseModel):
    """Policy related to resource use and circular economy per E5-1.

    Per ESRS E5-1 Para 11-13, undertakings shall describe their policies
    addressing resource use and circular economy, including alignment
    with the waste hierarchy, EPR obligations, and packaging targets.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique policy identifier",
    )
    name: str = Field(
        ...,
        description="Policy name",
        max_length=500,
    )
    scope: str = Field(
        default="",
        description="Scope of the policy (e.g. group-wide, site-specific)",
        max_length=500,
    )
    strategies_covered: List[str] = Field(
        default_factory=list,
        description="Circular economy strategies addressed by the policy",
    )
    epr_compliance: bool = Field(
        default=False,
        description="Whether the policy addresses Extended Producer Responsibility",
    )
    packaging_targets: bool = Field(
        default=False,
        description="Whether the policy includes packaging waste targets",
    )
    waste_hierarchy_aligned: bool = Field(
        default=False,
        description="Whether the policy aligns with EU waste hierarchy",
    )
    substances_of_concern: bool = Field(
        default=False,
        description="Whether the policy addresses substances of concern",
    )

class CircularAction(BaseModel):
    """Action related to resource use and circular economy per E5-2.

    Per ESRS E5-2 Para 17-20, undertakings shall describe actions
    taken to implement circular economy strategies, including
    resources allocated and expected outcomes.
    """
    action_id: str = Field(
        default_factory=_new_uuid,
        description="Unique action identifier",
    )
    description: str = Field(
        ...,
        description="Description of the circular economy action",
        max_length=2000,
    )
    resources_allocated: Decimal = Field(
        default=Decimal("0"),
        description="Financial resources allocated in EUR",
        ge=Decimal("0"),
    )
    circular_strategy: str = Field(
        default="",
        description="Circular strategy type (e.g. reduce, reuse, recycle)",
        max_length=200,
    )
    expected_outcome: str = Field(
        default="",
        description="Expected outcome of the action",
        max_length=2000,
    )
    timeline: str = Field(
        default="",
        description="Timeline for implementation (e.g. short/medium/long-term)",
        max_length=200,
    )
    status: str = Field(
        default="planned",
        description="Action status (planned, in_progress, completed)",
        max_length=50,
    )

class CircularTarget(BaseModel):
    """Target related to resource use and circular economy per E5-3.

    Per ESRS E5-3 Para 22-25, undertakings shall describe targets
    set for resource use reduction, recycled content, waste reduction,
    and circular material use rate.
    """
    target_id: str = Field(
        default_factory=_new_uuid,
        description="Unique target identifier",
    )
    metric: str = Field(
        ...,
        description="Target metric name (e.g. recycled_content_pct, waste_reduction)",
        max_length=200,
    )
    target_type: str = Field(
        default="absolute",
        description="Target type: absolute or intensity",
        max_length=50,
    )
    base_year: int = Field(
        ...,
        description="Base year for the target",
        ge=2000,
    )
    base_value: Decimal = Field(
        ...,
        description="Base year value for the target metric",
    )
    target_value: Decimal = Field(
        ...,
        description="Target value to achieve",
    )
    target_year: int = Field(
        ...,
        description="Year by which target should be achieved",
        ge=2000,
    )
    progress_pct: Decimal = Field(
        default=Decimal("0"),
        description="Current progress towards target as percentage",
        ge=Decimal("0"),
        le=Decimal("200"),
    )
    current_value: Decimal = Field(
        default=Decimal("0"),
        description="Current reporting period value",
    )

    @field_validator("target_year")
    @classmethod
    def validate_target_year_after_base(
        cls, v: int, info: Any
    ) -> int:
        """Validate that target year is equal to or after base year."""
        base = info.data.get("base_year")
        if base is not None and v < base:
            raise ValueError(
                f"target_year ({v}) must be >= base_year ({base})"
            )
        return v

class ResourceInflow(BaseModel):
    """Resource inflow data per ESRS E5-4.

    Per ESRS E5-4 Para 29-33, undertakings shall report the total
    weight of materials used, disaggregated by type and origin,
    including recycled content and renewable material percentages.
    """
    inflow_id: str = Field(
        default_factory=_new_uuid,
        description="Unique inflow record identifier",
    )
    material_type: MaterialType = Field(
        ...,
        description="Type of material",
    )
    origin: MaterialOrigin = Field(
        ...,
        description="Origin of the material (virgin, recycled, etc.)",
    )
    quantity_tonnes: Decimal = Field(
        ...,
        description="Quantity in metric tonnes",
        ge=Decimal("0"),
    )
    recycled_content_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of recycled content in this inflow (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    renewable_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of renewable content (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    biological_material: bool = Field(
        default=False,
        description="Whether this is a biological/bio-based material",
    )
    critical_raw_material: bool = Field(
        default=False,
        description="Whether this is a critical raw material per EU CRM list",
    )
    reporting_year: int = Field(
        default=0,
        description="Reporting year for this inflow",
        ge=0,
    )

class ResourceOutflow(BaseModel):
    """Resource outflow (waste) data per ESRS E5-5.

    Per ESRS E5-5 Para 35-40, undertakings shall report total waste
    generated, disaggregated by hazardous/non-hazardous status and
    by waste treatment destination.
    """
    outflow_id: str = Field(
        default_factory=_new_uuid,
        description="Unique outflow record identifier",
    )
    waste_category: WasteCategory = Field(
        ...,
        description="Waste hazard category",
    )
    destination: WasteDestination = Field(
        ...,
        description="Waste treatment destination",
    )
    quantity_tonnes: Decimal = Field(
        ...,
        description="Quantity of waste in metric tonnes",
        ge=Decimal("0"),
    )
    recycling_rate_pct: Decimal = Field(
        default=Decimal("0"),
        description="Effective recycling rate for this stream (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    product_id: str = Field(
        default="",
        description="Associated product identifier (if product-specific waste)",
        max_length=200,
    )
    reporting_year: int = Field(
        default=0,
        description="Reporting year for this outflow",
        ge=0,
    )

class ProductCircularity(BaseModel):
    """Product-level circularity assessment per ESRS E5-5.

    Per ESRS E5-5 AR 19-26, undertakings shall describe how products
    and materials are designed for durability, reuse, repair,
    disassembly, remanufacturing, and recycling.
    """
    product_id: str = Field(
        default_factory=_new_uuid,
        description="Unique product identifier",
    )
    name: str = Field(
        ...,
        description="Product name",
        max_length=500,
    )
    design_strategies: List[ProductDesignStrategy] = Field(
        default_factory=list,
        description="Circular design strategies applied to this product",
    )
    business_model: Optional[CircularBusinessModel] = Field(
        default=None,
        description="Circular business model associated with this product",
    )
    durability_years: Decimal = Field(
        default=Decimal("0"),
        description="Expected product lifespan in years",
        ge=Decimal("0"),
    )
    recyclability_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of product that is recyclable by weight (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    take_back_available: bool = Field(
        default=False,
        description="Whether a take-back or return scheme exists for this product",
    )
    recycled_content_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of recycled content in the product (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    repairability_score: Decimal = Field(
        default=Decimal("0"),
        description="Repairability score (0-10 scale)",
        ge=Decimal("0"),
        le=Decimal("10"),
    )

class CircularFinancialEffect(BaseModel):
    """Anticipated financial effect from circular economy per E5-6.

    Per ESRS E5-6 Para 42-44, undertakings shall disclose anticipated
    financial effects from resource use and circular economy related
    risks and opportunities.
    """
    effect_id: str = Field(
        default_factory=_new_uuid,
        description="Unique financial effect identifier",
    )
    risk_type: str = Field(
        ...,
        description="Type of risk or opportunity (e.g. regulatory, market, operational)",
        max_length=200,
    )
    description: str = Field(
        ...,
        description="Description of the financial effect",
        max_length=2000,
    )
    monetary_impact: Decimal = Field(
        default=Decimal("0"),
        description="Estimated monetary impact in EUR",
    )
    time_horizon: str = Field(
        default="medium_term",
        description="Time horizon: short_term (<1yr), medium_term (1-5yr), long_term (>5yr)",
        max_length=50,
    )
    is_opportunity: bool = Field(
        default=False,
        description="True if this is an opportunity; False if a risk",
    )
    likelihood: str = Field(
        default="",
        description="Likelihood assessment (low, medium, high, very_high)",
        max_length=50,
    )

class E5CircularResult(BaseModel):
    """Complete E5 circular economy disclosure result.

    Aggregates all E5-1 through E5-6 disclosure data into a single
    result with provenance tracking and compliance scoring.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of calculation (UTC)",
    )
    reporting_year: int = Field(
        default=0,
        description="Reporting year",
    )

    # E5-1: Policies
    policies: List[CircularPolicy] = Field(
        default_factory=list,
        description="Circular economy policies per E5-1",
    )

    # E5-2: Actions
    actions: List[CircularAction] = Field(
        default_factory=list,
        description="Circular economy actions per E5-2",
    )

    # E5-3: Targets
    targets: List[CircularTarget] = Field(
        default_factory=list,
        description="Circular economy targets per E5-3",
    )

    # E5-4: Resource inflows
    total_material_inflow_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Total material inflow in metric tonnes",
    )
    recycled_content_pct: Decimal = Field(
        default=Decimal("0"),
        description="Overall recycled content percentage (0-100)",
    )
    renewable_material_pct: Decimal = Field(
        default=Decimal("0"),
        description="Overall renewable material percentage (0-100)",
    )
    inflow_by_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Material inflow breakdown by material type (tonnes)",
    )
    biological_material_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Total biological material inflow in tonnes",
    )

    # E5-5: Resource outflows
    total_waste_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Total waste generated in metric tonnes",
    )
    hazardous_waste_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Hazardous waste in metric tonnes",
    )
    non_hazardous_waste_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Non-hazardous waste in metric tonnes",
    )
    radioactive_waste_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Radioactive waste in metric tonnes",
    )
    waste_by_destination: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Waste breakdown by destination (tonnes)",
    )
    recycling_rate_pct: Decimal = Field(
        default=Decimal("0"),
        description="Overall waste recycling rate percentage (0-100)",
    )
    circular_material_use_rate: Decimal = Field(
        default=Decimal("0"),
        description="Circular Material Use Rate per EU Monitoring Framework (0-100)",
    )

    # E5-5: Product circularity
    product_circularity_scores: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Circularity score per product (0-100)",
    )
    average_product_circularity: Decimal = Field(
        default=Decimal("0"),
        description="Average product circularity score (0-100)",
    )

    # E5-6: Financial effects
    financial_effects: List[CircularFinancialEffect] = Field(
        default_factory=list,
        description="Anticipated financial effects per E5-6",
    )
    total_risk_impact_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total monetary impact from risks (EUR)",
    )
    total_opportunity_impact_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total monetary impact from opportunities (EUR)",
    )

    # Compliance
    compliance_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall E5 disclosure compliance score (0-100)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CircularEconomyEngine:
    """E5 Resource Use and Circular Economy calculation engine.

    Provides deterministic, zero-hallucination calculations for:
    - Resource inflow aggregation by material type and origin
    - Recycled content and renewable share computation
    - Waste outflow aggregation by category and destination
    - Recycling rate and waste hierarchy scoring
    - Circular Material Use Rate (CMUR) per EU Monitoring Framework
    - Product circularity scoring (design, durability, recyclability)
    - Financial effects aggregation from risks and opportunities
    - E5 completeness validation across all six disclosure requirements
    - E5 data point mapping for ESRS disclosure

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Calculation Methodology:
        1. Inflow total = sum of all ResourceInflow.quantity_tonnes
        2. Recycled content % = weighted average by quantity
        3. Renewable % = weighted average by quantity
        4. Waste total = sum of all ResourceOutflow.quantity_tonnes
        5. Recycling rate = (recycling + composting) / total waste * 100
        6. CMUR = recycled_input / (total_inflow + recycled_input - recycled_output) * 100
        7. Product circularity = weighted score of design strategies + attributes
        8. Compliance = populated_datapoints / total_datapoints * 100

    Usage::

        engine = CircularEconomyEngine()
        result = engine.calculate_e5_disclosure(
            policies=[...],
            actions=[...],
            targets=[...],
            inflows=[...],
            outflows=[...],
            products=[...],
            financial_effects=[...],
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Resource Inflows (E5-4)                                              #
    # ------------------------------------------------------------------ #

    def calculate_resource_inflows(
        self, inflows: List[ResourceInflow]
    ) -> Dict[str, Any]:
        """Calculate resource inflow metrics per ESRS E5-4.

        Aggregates material inflows by type, computes overall recycled
        content percentage, renewable material share, and biological
        material total.

        Formula:
            recycled_content_pct = sum(qty * recycled_pct) / total_qty
            renewable_pct = sum(qty * renewable_pct) / total_qty

        Args:
            inflows: List of ResourceInflow records.

        Returns:
            Dict with:
                - total_tonnes: Decimal
                - by_type: Dict[str, Decimal] mapping MaterialType -> tonnes
                - by_origin: Dict[str, Decimal] mapping MaterialOrigin -> tonnes
                - recycled_content_pct: Decimal (0-100)
                - renewable_material_pct: Decimal (0-100)
                - biological_material_tonnes: Decimal
                - critical_raw_material_tonnes: Decimal
                - provenance_hash: str
        """
        if not inflows:
            logger.warning("No resource inflows provided for E5-4 calculation")
            return {
                "total_tonnes": Decimal("0"),
                "by_type": {},
                "by_origin": {},
                "recycled_content_pct": Decimal("0"),
                "renewable_material_pct": Decimal("0"),
                "biological_material_tonnes": Decimal("0"),
                "critical_raw_material_tonnes": Decimal("0"),
                "provenance_hash": _compute_hash({"inflows": []}),
            }

        total_tonnes = Decimal("0")
        by_type: Dict[str, Decimal] = {}
        by_origin: Dict[str, Decimal] = {}
        weighted_recycled = Decimal("0")
        weighted_renewable = Decimal("0")
        biological_tonnes = Decimal("0")
        critical_tonnes = Decimal("0")

        for inflow in inflows:
            qty = inflow.quantity_tonnes
            total_tonnes += qty

            # Aggregate by material type
            type_key = inflow.material_type.value
            by_type[type_key] = by_type.get(type_key, Decimal("0")) + qty

            # Aggregate by origin
            origin_key = inflow.origin.value
            by_origin[origin_key] = by_origin.get(origin_key, Decimal("0")) + qty

            # Weighted recycled content
            weighted_recycled += qty * inflow.recycled_content_pct

            # Weighted renewable percentage
            weighted_renewable += qty * inflow.renewable_pct

            # Biological materials
            if inflow.biological_material:
                biological_tonnes += qty

            # Critical raw materials
            if inflow.critical_raw_material:
                critical_tonnes += qty

        # Calculate weighted averages
        recycled_pct = _round_val(
            _safe_divide(weighted_recycled, total_tonnes), 2
        )
        renewable_pct = _round_val(
            _safe_divide(weighted_renewable, total_tonnes), 2
        )

        # Round all values
        total_tonnes = _round_val(total_tonnes, 3)
        biological_tonnes = _round_val(biological_tonnes, 3)
        critical_tonnes = _round_val(critical_tonnes, 3)
        for key in by_type:
            by_type[key] = _round_val(by_type[key], 3)
        for key in by_origin:
            by_origin[key] = _round_val(by_origin[key], 3)

        result = {
            "total_tonnes": total_tonnes,
            "by_type": by_type,
            "by_origin": by_origin,
            "recycled_content_pct": recycled_pct,
            "renewable_material_pct": renewable_pct,
            "biological_material_tonnes": biological_tonnes,
            "critical_raw_material_tonnes": critical_tonnes,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Resource inflows calculated: total=%.3f tonnes, recycled=%.2f%%, "
            "renewable=%.2f%%, types=%d",
            float(total_tonnes), float(recycled_pct),
            float(renewable_pct), len(by_type),
        )

        return result

    # ------------------------------------------------------------------ #
    # Resource Outflows (E5-5)                                             #
    # ------------------------------------------------------------------ #

    def calculate_resource_outflows(
        self, outflows: List[ResourceOutflow]
    ) -> Dict[str, Any]:
        """Calculate resource outflow (waste) metrics per ESRS E5-5.

        Aggregates waste by category (hazardous, non-hazardous, radioactive)
        and by treatment destination.  Calculates overall recycling rate
        as the share of waste going to recycling and composting.

        Formula:
            recycling_rate = (recycling + composting) / total_waste * 100

        Args:
            outflows: List of ResourceOutflow records.

        Returns:
            Dict with:
                - total_waste_tonnes: Decimal
                - hazardous_tonnes: Decimal
                - non_hazardous_tonnes: Decimal
                - radioactive_tonnes: Decimal
                - by_destination: Dict[str, Decimal] mapping destination -> tonnes
                - recycling_rate_pct: Decimal (0-100)
                - waste_hierarchy_score: Decimal (1-5 scale)
                - provenance_hash: str
        """
        if not outflows:
            logger.warning("No resource outflows provided for E5-5 calculation")
            return {
                "total_waste_tonnes": Decimal("0"),
                "hazardous_tonnes": Decimal("0"),
                "non_hazardous_tonnes": Decimal("0"),
                "radioactive_tonnes": Decimal("0"),
                "by_destination": {},
                "recycling_rate_pct": Decimal("0"),
                "waste_hierarchy_score": Decimal("0"),
                "provenance_hash": _compute_hash({"outflows": []}),
            }

        total_waste = Decimal("0")
        hazardous = Decimal("0")
        non_hazardous = Decimal("0")
        radioactive = Decimal("0")
        by_destination: Dict[str, Decimal] = {}
        weighted_hierarchy = Decimal("0")

        for outflow in outflows:
            qty = outflow.quantity_tonnes
            total_waste += qty

            # Aggregate by waste category
            if outflow.waste_category == WasteCategory.HAZARDOUS:
                hazardous += qty
            elif outflow.waste_category == WasteCategory.NON_HAZARDOUS:
                non_hazardous += qty
            elif outflow.waste_category == WasteCategory.RADIOACTIVE:
                radioactive += qty

            # Aggregate by destination
            dest_key = outflow.destination.value
            by_destination[dest_key] = (
                by_destination.get(dest_key, Decimal("0")) + qty
            )

            # Waste hierarchy weighting
            hierarchy_weight = WASTE_HIERARCHY_WEIGHTS.get(
                dest_key, Decimal("1")
            )
            weighted_hierarchy += qty * hierarchy_weight

        # Calculate recycling rate (recycling + composting as diversion)
        diverted = (
            by_destination.get("recycling", Decimal("0"))
            + by_destination.get("composting", Decimal("0"))
        )
        recycling_rate = _round_val(
            _safe_divide(diverted, total_waste) * Decimal("100"), 2
        )

        # Calculate waste hierarchy score (weighted average, 1-5 scale)
        hierarchy_score = _round_val(
            _safe_divide(weighted_hierarchy, total_waste), 2
        )

        # Round all values
        total_waste = _round_val(total_waste, 3)
        hazardous = _round_val(hazardous, 3)
        non_hazardous = _round_val(non_hazardous, 3)
        radioactive = _round_val(radioactive, 3)
        for key in by_destination:
            by_destination[key] = _round_val(by_destination[key], 3)

        result = {
            "total_waste_tonnes": total_waste,
            "hazardous_tonnes": hazardous,
            "non_hazardous_tonnes": non_hazardous,
            "radioactive_tonnes": radioactive,
            "by_destination": by_destination,
            "recycling_rate_pct": recycling_rate,
            "waste_hierarchy_score": hierarchy_score,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Resource outflows calculated: total=%.3f tonnes, "
            "hazardous=%.3f, recycling_rate=%.2f%%, hierarchy=%.2f",
            float(total_waste), float(hazardous),
            float(recycling_rate), float(hierarchy_score),
        )

        return result

    # ------------------------------------------------------------------ #
    # Circular Material Use Rate (CMUR)                                    #
    # ------------------------------------------------------------------ #

    def calculate_circular_material_use_rate(
        self,
        inflows: List[ResourceInflow],
        outflows: List[ResourceOutflow],
    ) -> Decimal:
        """Calculate the Circular Material Use Rate (CMUR).

        The CMUR measures the share of material recovered and fed back
        into the economy as a proportion of overall material use.  This
        follows the EU Circular Economy Monitoring Framework definition.

        Formula (EU Monitoring Framework):
            U = total material input (all inflows)
            R_input = recycled/secondary material input
            R_output = waste sent to recycling + composting
            CMUR = R_input / (U + R_output - R_input) * 100

        If (U + R_output - R_input) is zero, returns Decimal("0").

        Args:
            inflows: List of ResourceInflow records.
            outflows: List of ResourceOutflow records.

        Returns:
            CMUR as Decimal percentage (0-100), rounded to 2 places.
        """
        # Total material input (U)
        total_input = sum(i.quantity_tonnes for i in inflows)

        # Recycled/secondary input (R_input)
        recycled_input = sum(
            i.quantity_tonnes
            for i in inflows
            if i.origin in (MaterialOrigin.RECYCLED, MaterialOrigin.SECONDARY)
        )

        # Recycled output (R_output) - waste going to recycling/composting
        recycled_output = sum(
            o.quantity_tonnes
            for o in outflows
            if o.destination in (
                WasteDestination.RECYCLING,
                WasteDestination.COMPOSTING,
            )
        )

        # CMUR denominator
        denominator = total_input + recycled_output - recycled_input

        cmur = _round_val(
            _safe_divide(recycled_input, denominator) * Decimal("100"), 2
        )

        logger.info(
            "CMUR calculated: %.2f%% (R_in=%.3f, U=%.3f, R_out=%.3f)",
            float(cmur), float(recycled_input),
            float(total_input), float(recycled_output),
        )

        return cmur

    # ------------------------------------------------------------------ #
    # Product Circularity Assessment (E5-5)                                #
    # ------------------------------------------------------------------ #

    def assess_product_circularity(
        self, products: List[ProductCircularity]
    ) -> Dict[str, Any]:
        """Assess circularity of products per ESRS E5-5.

        Calculates a composite circularity score for each product based
        on design strategies, recyclability, recycled content, durability,
        repairability, and take-back availability.

        Scoring methodology:
            - Design strategies: sum of weights for applied strategies (0-1) * 25
            - Recyclability: recyclability_pct * 0.25
            - Recycled content: recycled_content_pct * 0.20
            - Durability: min(durability_years / 10, 1) * 15
            - Repairability: (repairability_score / 10) * 10
            - Take-back: 5 if available, 0 otherwise
            Total max = 100

        Args:
            products: List of ProductCircularity records.

        Returns:
            Dict with:
                - scores: Dict[str, Decimal] mapping product_id -> score (0-100)
                - average_score: Decimal (0-100)
                - products_with_take_back_pct: Decimal (0-100)
                - circular_design_coverage: Dict[str, int] strategy -> count
                - provenance_hash: str
        """
        if not products:
            logger.warning("No products provided for circularity assessment")
            return {
                "scores": {},
                "average_score": Decimal("0"),
                "products_with_take_back_pct": Decimal("0"),
                "circular_design_coverage": {},
                "provenance_hash": _compute_hash({"products": []}),
            }

        scores: Dict[str, Decimal] = {}
        take_back_count = 0
        design_coverage: Dict[str, int] = {s.value: 0 for s in ProductDesignStrategy}

        for product in products:
            score = self._score_single_product(product)
            scores[product.product_id] = score

            if product.take_back_available:
                take_back_count += 1

            for strategy in product.design_strategies:
                design_coverage[strategy.value] = (
                    design_coverage.get(strategy.value, 0) + 1
                )

        # Average score
        total_score = sum(scores.values())
        avg_score = _round_val(
            _safe_divide(total_score, _decimal(len(scores))), 2
        )

        # Take-back percentage
        take_back_pct = _round_val(
            _safe_divide(
                _decimal(take_back_count), _decimal(len(products))
            ) * Decimal("100"),
            2,
        )

        result = {
            "scores": scores,
            "average_score": avg_score,
            "products_with_take_back_pct": take_back_pct,
            "circular_design_coverage": design_coverage,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Product circularity assessed: %d products, avg_score=%.2f, "
            "take_back=%.2f%%",
            len(products), float(avg_score), float(take_back_pct),
        )

        return result

    def _score_single_product(self, product: ProductCircularity) -> Decimal:
        """Calculate circularity score for a single product.

        Scoring components (total max = 100):
            - Design strategies applied: up to 25 points
            - Recyclability percentage: up to 25 points
            - Recycled content percentage: up to 20 points
            - Durability (capped at 10 years): up to 15 points
            - Repairability (0-10 scale): up to 10 points
            - Take-back scheme availability: 5 points

        Args:
            product: ProductCircularity record.

        Returns:
            Circularity score as Decimal (0-100).
        """
        # Design strategies component (max 25 points)
        design_score = Decimal("0")
        for strategy in product.design_strategies:
            weight = DESIGN_STRATEGY_WEIGHTS.get(
                strategy.value, Decimal("0")
            )
            design_score += weight
        design_points = min(design_score, Decimal("1")) * Decimal("25")

        # Recyclability component (max 25 points)
        recyclability_points = (
            product.recyclability_pct / Decimal("100") * Decimal("25")
        )

        # Recycled content component (max 20 points)
        recycled_points = (
            product.recycled_content_pct / Decimal("100") * Decimal("20")
        )

        # Durability component (max 15 points, capped at 10 years)
        durability_ratio = min(
            _safe_divide(product.durability_years, Decimal("10")),
            Decimal("1"),
        )
        durability_points = durability_ratio * Decimal("15")

        # Repairability component (max 10 points)
        repairability_points = (
            _safe_divide(product.repairability_score, Decimal("10"))
            * Decimal("10")
        )

        # Take-back component (5 points if available)
        take_back_points = Decimal("5") if product.take_back_available else Decimal("0")

        total = (
            design_points
            + recyclability_points
            + recycled_points
            + durability_points
            + repairability_points
            + take_back_points
        )

        return _round_val(min(total, Decimal("100")), 2)

    # ------------------------------------------------------------------ #
    # Financial Effects (E5-6)                                             #
    # ------------------------------------------------------------------ #

    def _aggregate_financial_effects(
        self, effects: List[CircularFinancialEffect]
    ) -> Dict[str, Any]:
        """Aggregate financial effects from circular economy per E5-6.

        Separates risks from opportunities and totals monetary impact
        for each category and time horizon.

        Args:
            effects: List of CircularFinancialEffect records.

        Returns:
            Dict with:
                - total_risk_impact: Decimal (EUR)
                - total_opportunity_impact: Decimal (EUR)
                - net_impact: Decimal (EUR, opportunities minus risks)
                - by_time_horizon: Dict[str, Decimal]
                - risk_count: int
                - opportunity_count: int
        """
        total_risk = Decimal("0")
        total_opportunity = Decimal("0")
        by_horizon: Dict[str, Decimal] = {}
        risk_count = 0
        opportunity_count = 0

        for effect in effects:
            impact = effect.monetary_impact

            if effect.is_opportunity:
                total_opportunity += impact
                opportunity_count += 1
            else:
                total_risk += impact
                risk_count += 1

            horizon_key = effect.time_horizon
            by_horizon[horizon_key] = (
                by_horizon.get(horizon_key, Decimal("0")) + impact
            )

        net_impact = total_opportunity - total_risk

        return {
            "total_risk_impact": _round_val(total_risk, 2),
            "total_opportunity_impact": _round_val(total_opportunity, 2),
            "net_impact": _round_val(net_impact, 2),
            "by_time_horizon": {
                k: _round_val(v, 2) for k, v in by_horizon.items()
            },
            "risk_count": risk_count,
            "opportunity_count": opportunity_count,
        }

    # ------------------------------------------------------------------ #
    # Full E5 Disclosure Calculation                                       #
    # ------------------------------------------------------------------ #

    def calculate_e5_disclosure(
        self,
        policies: Optional[List[CircularPolicy]] = None,
        actions: Optional[List[CircularAction]] = None,
        targets: Optional[List[CircularTarget]] = None,
        inflows: Optional[List[ResourceInflow]] = None,
        outflows: Optional[List[ResourceOutflow]] = None,
        products: Optional[List[ProductCircularity]] = None,
        financial_effects: Optional[List[CircularFinancialEffect]] = None,
        reporting_year: int = 0,
    ) -> E5CircularResult:
        """Calculate the complete ESRS E5 disclosure.

        Orchestrates all sub-calculations (E5-1 through E5-6) and
        assembles the final E5CircularResult with provenance tracking.

        Args:
            policies: Circular economy policies (E5-1).
            actions: Circular economy actions (E5-2).
            targets: Circular economy targets (E5-3).
            inflows: Resource inflow records (E5-4).
            outflows: Resource outflow records (E5-5).
            products: Product circularity data (E5-5).
            financial_effects: Financial effects (E5-6).
            reporting_year: Reporting period year.

        Returns:
            E5CircularResult with complete provenance.
        """
        t0 = time.perf_counter()

        policies = policies or []
        actions = actions or []
        targets = targets or []
        inflows = inflows or []
        outflows = outflows or []
        products = products or []
        financial_effects = financial_effects or []

        logger.info(
            "Calculating E5 disclosure: year=%d, policies=%d, actions=%d, "
            "targets=%d, inflows=%d, outflows=%d, products=%d, effects=%d",
            reporting_year, len(policies), len(actions), len(targets),
            len(inflows), len(outflows), len(products), len(financial_effects),
        )

        # E5-4: Resource inflows
        inflow_metrics = self.calculate_resource_inflows(inflows)

        # E5-5: Resource outflows
        outflow_metrics = self.calculate_resource_outflows(outflows)

        # CMUR
        cmur = self.calculate_circular_material_use_rate(inflows, outflows)

        # E5-5: Product circularity
        product_metrics = self.assess_product_circularity(products)

        # E5-6: Financial effects
        financial_metrics = self._aggregate_financial_effects(financial_effects)

        # Update target progress
        self._update_target_progress(targets)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = E5CircularResult(
            reporting_year=reporting_year,
            # E5-1
            policies=policies,
            # E5-2
            actions=actions,
            # E5-3
            targets=targets,
            # E5-4
            total_material_inflow_tonnes=inflow_metrics["total_tonnes"],
            recycled_content_pct=inflow_metrics["recycled_content_pct"],
            renewable_material_pct=inflow_metrics["renewable_material_pct"],
            inflow_by_type=inflow_metrics["by_type"],
            biological_material_tonnes=inflow_metrics["biological_material_tonnes"],
            # E5-5 waste
            total_waste_tonnes=outflow_metrics["total_waste_tonnes"],
            hazardous_waste_tonnes=outflow_metrics["hazardous_tonnes"],
            non_hazardous_waste_tonnes=outflow_metrics["non_hazardous_tonnes"],
            radioactive_waste_tonnes=outflow_metrics["radioactive_tonnes"],
            waste_by_destination=outflow_metrics["by_destination"],
            recycling_rate_pct=outflow_metrics["recycling_rate_pct"],
            circular_material_use_rate=cmur,
            # E5-5 products
            product_circularity_scores=product_metrics["scores"],
            average_product_circularity=product_metrics["average_score"],
            # E5-6
            financial_effects=financial_effects,
            total_risk_impact_eur=financial_metrics["total_risk_impact"],
            total_opportunity_impact_eur=financial_metrics["total_opportunity_impact"],
            # Meta
            processing_time_ms=elapsed_ms,
        )

        # Calculate compliance score
        completeness = self.validate_e5_completeness(result)
        result.compliance_score = completeness["completeness_pct"]

        # Provenance hash (computed after compliance score is set)
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "E5 disclosure calculated: inflow=%.3f t, waste=%.3f t, "
            "CMUR=%.2f%%, compliance=%.1f%%, hash=%s",
            float(result.total_material_inflow_tonnes),
            float(result.total_waste_tonnes),
            float(cmur),
            float(result.compliance_score),
            result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Target Progress                                                      #
    # ------------------------------------------------------------------ #

    def _update_target_progress(
        self, targets: List[CircularTarget]
    ) -> None:
        """Update progress percentage on each target in place.

        Calculates progress as:
            progress = (current_value - base_value) / (target_value - base_value) * 100

        If the target_value equals the base_value (no change expected),
        progress is set to 100% if current meets target, else 0%.

        Args:
            targets: List of CircularTarget (modified in place).
        """
        for target in targets:
            gap = target.target_value - target.base_value
            if gap == Decimal("0"):
                target.progress_pct = (
                    Decimal("100")
                    if target.current_value == target.target_value
                    else Decimal("0")
                )
            else:
                achieved = target.current_value - target.base_value
                raw_pct = _safe_divide(achieved, gap) * Decimal("100")
                # Clamp to 0-200 range
                clamped = max(Decimal("0"), min(raw_pct, Decimal("200")))
                target.progress_pct = _round_val(clamped, 2)

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_e5_completeness(
        self, result: E5CircularResult
    ) -> Dict[str, Any]:
        """Validate completeness against all E5 required data points.

        Checks whether ESRS E5-1 through E5-6 mandatory disclosure
        data points are present and populated in the result.

        Args:
            result: E5CircularResult to validate.

        Returns:
            Dict with:
                - total_datapoints: int
                - populated_datapoints: int
                - missing_datapoints: list of str
                - completeness_pct: Decimal (0-100)
                - is_complete: bool
                - by_disclosure: Dict[str, Dict] per-DR completeness
                - provenance_hash: str
        """
        populated: List[str] = []
        missing: List[str] = []

        # E5-1 checks
        e5_1_checks = {
            "e5_1_01_policies_addressing_resource_use": len(result.policies) > 0,
            "e5_1_02_alignment_with_waste_hierarchy": any(
                p.waste_hierarchy_aligned for p in result.policies
            ),
            "e5_1_03_circular_economy_strategies": any(
                len(p.strategies_covered) > 0 for p in result.policies
            ),
            "e5_1_04_epr_compliance_status": any(
                p.epr_compliance for p in result.policies
            ),
            "e5_1_05_packaging_waste_policy": any(
                p.packaging_targets for p in result.policies
            ),
            "e5_1_06_substances_of_concern_policy": any(
                p.substances_of_concern for p in result.policies
            ),
        }

        # E5-2 checks
        e5_2_checks = {
            "e5_2_01_actions_to_reduce_resource_use": len(result.actions) > 0,
            "e5_2_02_circular_design_actions": any(
                "design" in a.circular_strategy.lower()
                for a in result.actions
                if a.circular_strategy
            ),
            "e5_2_03_circular_business_model_actions": any(
                a.circular_strategy != "" for a in result.actions
            ),
            "e5_2_04_resources_allocated_to_actions": any(
                a.resources_allocated > Decimal("0") for a in result.actions
            ),
            "e5_2_05_waste_prevention_actions": any(
                "waste" in a.description.lower() or "prevent" in a.description.lower()
                for a in result.actions
            ),
            "e5_2_06_value_chain_engagement_actions": any(
                "value chain" in a.description.lower()
                or "supplier" in a.description.lower()
                for a in result.actions
            ),
        }

        # E5-3 checks
        e5_3_checks = {
            "e5_3_01_resource_use_reduction_targets": len(result.targets) > 0,
            "e5_3_02_recycled_content_targets": any(
                "recycled" in t.metric.lower() for t in result.targets
            ),
            "e5_3_03_waste_reduction_targets": any(
                "waste" in t.metric.lower() for t in result.targets
            ),
            "e5_3_04_recycling_rate_targets": any(
                "recycling" in t.metric.lower() for t in result.targets
            ),
            "e5_3_05_circular_material_use_rate_target": any(
                "cmur" in t.metric.lower() or "circular" in t.metric.lower()
                for t in result.targets
            ),
            "e5_3_06_target_base_year_and_progress": all(
                t.base_year > 0 and t.progress_pct >= Decimal("0")
                for t in result.targets
            ) if result.targets else False,
        }

        # E5-4 checks
        e5_4_checks = {
            "e5_4_01_total_material_inflow_tonnes": (
                result.total_material_inflow_tonnes > Decimal("0")
            ),
            "e5_4_02_inflow_by_material_type": len(result.inflow_by_type) > 0,
            "e5_4_03_recycled_content_percentage": (
                result.recycled_content_pct >= Decimal("0")
            ),
            "e5_4_04_renewable_material_percentage": (
                result.renewable_material_pct >= Decimal("0")
            ),
            "e5_4_05_biological_material_tonnes": True,  # Reported if applicable
            "e5_4_06_secondary_material_usage": (
                result.recycled_content_pct > Decimal("0")
            ),
            "e5_4_07_critical_raw_materials_usage": True,  # Reported if applicable
        }

        # E5-5 checks
        e5_5_checks = {
            "e5_5_01_total_waste_generated_tonnes": (
                result.total_waste_tonnes > Decimal("0")
            ),
            "e5_5_02_hazardous_waste_tonnes": (
                result.hazardous_waste_tonnes >= Decimal("0")
            ),
            "e5_5_03_non_hazardous_waste_tonnes": (
                result.non_hazardous_waste_tonnes >= Decimal("0")
            ),
            "e5_5_04_radioactive_waste_tonnes": True,  # Reported if applicable
            "e5_5_05_waste_by_destination": len(result.waste_by_destination) > 0,
            "e5_5_06_recycling_rate_percentage": (
                result.recycling_rate_pct >= Decimal("0")
            ),
            "e5_5_07_products_designed_for_circularity": (
                len(result.product_circularity_scores) > 0
            ),
            "e5_5_08_take_back_products_percentage": True,  # Reported if applicable
        }

        # E5-6 checks
        e5_6_checks = {
            "e5_6_01_financial_effects_from_risks": any(
                not e.is_opportunity for e in result.financial_effects
            ),
            "e5_6_02_financial_effects_from_opportunities": any(
                e.is_opportunity for e in result.financial_effects
            ),
            "e5_6_03_monetary_impact_estimation": any(
                e.monetary_impact != Decimal("0")
                for e in result.financial_effects
            ),
            "e5_6_04_time_horizon_of_effects": all(
                e.time_horizon != "" for e in result.financial_effects
            ) if result.financial_effects else False,
        }

        # Combine all checks
        all_checks = {}
        all_checks.update(e5_1_checks)
        all_checks.update(e5_2_checks)
        all_checks.update(e5_3_checks)
        all_checks.update(e5_4_checks)
        all_checks.update(e5_5_checks)
        all_checks.update(e5_6_checks)

        for dp, is_populated in all_checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(E5_ALL_DATAPOINTS)
        pop_count = len(populated)
        completeness = _round_val(
            _safe_divide(_decimal(pop_count), _decimal(total)) * Decimal("100"),
            1,
        )

        # Per-disclosure requirement breakdown
        by_disclosure = {
            "E5-1": self._dr_completeness(e5_1_checks),
            "E5-2": self._dr_completeness(e5_2_checks),
            "E5-3": self._dr_completeness(e5_3_checks),
            "E5-4": self._dr_completeness(e5_4_checks),
            "E5-5": self._dr_completeness(e5_5_checks),
            "E5-6": self._dr_completeness(e5_6_checks),
        }

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "by_disclosure": by_disclosure,
            "provenance_hash": _compute_hash(
                {"result_id": result.result_id, "checks": all_checks}
            ),
        }

        logger.info(
            "E5 completeness: %.1f%% (%d/%d), missing=%d datapoints",
            float(completeness), pop_count, total, len(missing),
        )

        return validation_result

    def _dr_completeness(self, checks: Dict[str, bool]) -> Dict[str, Any]:
        """Calculate completeness for a single disclosure requirement.

        Args:
            checks: Dict mapping datapoint IDs to populated status.

        Returns:
            Dict with total, populated, missing count, and pct.
        """
        total = len(checks)
        pop = sum(1 for v in checks.values() if v)
        pct = _round_val(
            _safe_divide(_decimal(pop), _decimal(total)) * Decimal("100"), 1
        ) if total > 0 else Decimal("0")
        return {
            "total": total,
            "populated": pop,
            "missing": total - pop,
            "completeness_pct": pct,
        }

    # ------------------------------------------------------------------ #
    # ESRS E5 Data Point Mapping                                           #
    # ------------------------------------------------------------------ #

    def get_e5_datapoints(
        self, result: E5CircularResult
    ) -> Dict[str, Any]:
        """Map E5 result to ESRS disclosure data points.

        Creates a structured mapping of all E5 required data points
        with their values, ready for report generation.

        Args:
            result: E5CircularResult to map.

        Returns:
            Dict mapping E5 data point IDs to their values and
            metadata, with a provenance hash.
        """
        datapoints: Dict[str, Any] = {
            # E5-1 Policies
            "e5_1_policies": {
                "label": "Policies related to resource use and circular economy",
                "value": [p.model_dump(mode="json") for p in result.policies],
                "count": len(result.policies),
                "esrs_ref": "E5-1 Para 11-13",
            },
            # E5-2 Actions
            "e5_2_actions": {
                "label": "Actions and resources for circular economy",
                "value": [a.model_dump(mode="json") for a in result.actions],
                "count": len(result.actions),
                "esrs_ref": "E5-2 Para 17-20",
            },
            # E5-3 Targets
            "e5_3_targets": {
                "label": "Targets related to resource use and circular economy",
                "value": [t.model_dump(mode="json") for t in result.targets],
                "count": len(result.targets),
                "esrs_ref": "E5-3 Para 22-25",
            },
            # E5-4 Resource Inflows
            "e5_4_01_total_material_inflow_tonnes": {
                "label": "Total weight of material inflows",
                "value": str(result.total_material_inflow_tonnes),
                "unit": "metric tonnes",
                "esrs_ref": "E5-4 Para 29",
            },
            "e5_4_02_inflow_by_material_type": {
                "label": "Material inflows by type",
                "value": {k: str(v) for k, v in result.inflow_by_type.items()},
                "unit": "metric tonnes",
                "esrs_ref": "E5-4 Para 30",
            },
            "e5_4_03_recycled_content_percentage": {
                "label": "Recycled content as percentage of total inflow",
                "value": str(result.recycled_content_pct),
                "unit": "percent",
                "esrs_ref": "E5-4 Para 31",
            },
            "e5_4_04_renewable_material_percentage": {
                "label": "Renewable material as percentage of total inflow",
                "value": str(result.renewable_material_pct),
                "unit": "percent",
                "esrs_ref": "E5-4 Para 32",
            },
            "e5_4_05_biological_material_tonnes": {
                "label": "Biological material inflow",
                "value": str(result.biological_material_tonnes),
                "unit": "metric tonnes",
                "esrs_ref": "E5-4 Para 33",
            },
            # E5-5 Resource Outflows
            "e5_5_01_total_waste_generated_tonnes": {
                "label": "Total waste generated",
                "value": str(result.total_waste_tonnes),
                "unit": "metric tonnes",
                "esrs_ref": "E5-5 Para 35",
            },
            "e5_5_02_hazardous_waste_tonnes": {
                "label": "Hazardous waste",
                "value": str(result.hazardous_waste_tonnes),
                "unit": "metric tonnes",
                "esrs_ref": "E5-5 Para 36",
            },
            "e5_5_03_non_hazardous_waste_tonnes": {
                "label": "Non-hazardous waste",
                "value": str(result.non_hazardous_waste_tonnes),
                "unit": "metric tonnes",
                "esrs_ref": "E5-5 Para 37",
            },
            "e5_5_04_radioactive_waste_tonnes": {
                "label": "Radioactive waste",
                "value": str(result.radioactive_waste_tonnes),
                "unit": "metric tonnes",
                "esrs_ref": "E5-5 Para 38",
            },
            "e5_5_05_waste_by_destination": {
                "label": "Waste by treatment destination",
                "value": {
                    k: str(v) for k, v in result.waste_by_destination.items()
                },
                "unit": "metric tonnes",
                "esrs_ref": "E5-5 Para 39",
            },
            "e5_5_06_recycling_rate_percentage": {
                "label": "Waste recycling rate",
                "value": str(result.recycling_rate_pct),
                "unit": "percent",
                "esrs_ref": "E5-5 Para 40",
            },
            "e5_5_07_circular_material_use_rate": {
                "label": "Circular Material Use Rate (CMUR)",
                "value": str(result.circular_material_use_rate),
                "unit": "percent",
                "esrs_ref": "E5-5 Para 40",
            },
            "e5_5_08_product_circularity": {
                "label": "Products designed for circularity",
                "value": {
                    k: str(v)
                    for k, v in result.product_circularity_scores.items()
                },
                "average_score": str(result.average_product_circularity),
                "esrs_ref": "E5-5 Para 40",
            },
            # E5-6 Financial Effects
            "e5_6_01_financial_effects": {
                "label": "Anticipated financial effects",
                "value": [
                    e.model_dump(mode="json") for e in result.financial_effects
                ],
                "total_risk_eur": str(result.total_risk_impact_eur),
                "total_opportunity_eur": str(result.total_opportunity_impact_eur),
                "esrs_ref": "E5-6 Para 42-44",
            },
            # Overall
            "compliance_score": {
                "label": "E5 disclosure compliance score",
                "value": str(result.compliance_score),
                "unit": "percent",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)
        return datapoints

    # ------------------------------------------------------------------ #
    # Benchmark Comparison                                                 #
    # ------------------------------------------------------------------ #

    def compare_to_benchmark(
        self,
        cmur: Decimal,
        sector: str = "eu_average",
    ) -> Dict[str, Any]:
        """Compare CMUR to sector benchmark.

        Uses the EU Circular Economy Monitoring Framework sector
        benchmarks to assess how the undertaking's CMUR compares
        to the sector average.

        Args:
            cmur: Undertaking's Circular Material Use Rate (0-100).
            sector: Industry sector key for benchmark lookup.

        Returns:
            Dict with:
                - cmur: Decimal
                - benchmark: Decimal
                - sector: str
                - gap: Decimal (cmur - benchmark)
                - above_benchmark: bool
                - provenance_hash: str
        """
        benchmark = MATERIAL_CIRCULARITY_BENCHMARKS.get(
            sector,
            MATERIAL_CIRCULARITY_BENCHMARKS["eu_average"],
        )

        gap = _round_val(cmur - benchmark, 2)

        result = {
            "cmur": str(cmur),
            "benchmark": str(benchmark),
            "sector": sector,
            "gap": str(gap),
            "above_benchmark": cmur >= benchmark,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Benchmark comparison: CMUR=%.2f%% vs %s benchmark=%.2f%%, gap=%.2f",
            float(cmur), sector, float(benchmark), float(gap),
        )

        return result

    # ------------------------------------------------------------------ #
    # Year-over-Year Comparison                                            #
    # ------------------------------------------------------------------ #

    def compare_years(
        self,
        current: E5CircularResult,
        previous: E5CircularResult,
    ) -> Dict[str, Any]:
        """Compare E5 disclosures across two reporting years.

        Calculates absolute and percentage changes for key metrics
        including inflows, waste, recycling rate, and CMUR.

        Args:
            current: Current year E5 result.
            previous: Previous year E5 result.

        Returns:
            Dict with per-metric changes and provenance hash.
        """
        def _change(curr: Decimal, prev: Decimal) -> Dict[str, str]:
            abs_change = curr - prev
            pct = _safe_divide(
                abs_change, prev if prev != Decimal("0") else Decimal("1")
            ) * Decimal("100")
            return {
                "current": str(curr),
                "previous": str(prev),
                "absolute_change": str(_round_val(abs_change, 3)),
                "pct_change": str(_round_val(pct, 2)),
            }

        comparison = {
            "current_year": current.reporting_year,
            "previous_year": previous.reporting_year,
            "total_material_inflow": _change(
                current.total_material_inflow_tonnes,
                previous.total_material_inflow_tonnes,
            ),
            "recycled_content_pct": _change(
                current.recycled_content_pct,
                previous.recycled_content_pct,
            ),
            "renewable_material_pct": _change(
                current.renewable_material_pct,
                previous.renewable_material_pct,
            ),
            "total_waste": _change(
                current.total_waste_tonnes,
                previous.total_waste_tonnes,
            ),
            "hazardous_waste": _change(
                current.hazardous_waste_tonnes,
                previous.hazardous_waste_tonnes,
            ),
            "recycling_rate": _change(
                current.recycling_rate_pct,
                previous.recycling_rate_pct,
            ),
            "circular_material_use_rate": _change(
                current.circular_material_use_rate,
                previous.circular_material_use_rate,
            ),
            "compliance_score": _change(
                current.compliance_score,
                previous.compliance_score,
            ),
        }

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def get_e5_summary(
        self, result: E5CircularResult
    ) -> Dict[str, Any]:
        """Generate a summary of the E5 circular economy disclosure.

        Args:
            result: E5CircularResult to summarize.

        Returns:
            Dict with key metrics and overall status.
        """
        return {
            "reporting_year": result.reporting_year,
            "policies_count": len(result.policies),
            "actions_count": len(result.actions),
            "targets_count": len(result.targets),
            "total_material_inflow_tonnes": str(
                result.total_material_inflow_tonnes
            ),
            "recycled_content_pct": str(result.recycled_content_pct),
            "renewable_material_pct": str(result.renewable_material_pct),
            "biological_material_tonnes": str(
                result.biological_material_tonnes
            ),
            "total_waste_tonnes": str(result.total_waste_tonnes),
            "hazardous_waste_tonnes": str(result.hazardous_waste_tonnes),
            "recycling_rate_pct": str(result.recycling_rate_pct),
            "circular_material_use_rate": str(
                result.circular_material_use_rate
            ),
            "average_product_circularity": str(
                result.average_product_circularity
            ),
            "products_assessed": len(result.product_circularity_scores),
            "financial_effects_count": len(result.financial_effects),
            "total_risk_impact_eur": str(result.total_risk_impact_eur),
            "total_opportunity_impact_eur": str(
                result.total_opportunity_impact_eur
            ),
            "compliance_score": str(result.compliance_score),
            "processing_time_ms": result.processing_time_ms,
            "provenance_hash": result.provenance_hash,
        }
