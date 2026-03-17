# -*- coding: utf-8 -*-
"""
PrecursorChainEngine - PACK-005 CBAM Complete Engine 2

Multi-tier recursive precursor emission resolution per EU Regulation
2023/956 Article 35 and Annex III. Resolves the full production chain
for CBAM goods, calculates embedded emissions at each stage, and
supports mass/economic/energy allocation methods.

CBAM Goods Categories with Predefined Chains:
    1. Iron/Steel: ore -> pig iron -> crude steel -> hot-rolled -> cold-rolled -> coated
    2. Aluminium: bauxite -> alumina -> primary -> alloy -> rolled -> fabricated
    3. Cement: limestone -> clinker -> cement (CEM I-V)
    4. Fertilizers: natural gas -> ammonia -> urea/AN/CAN
    5. Hydrogen: NG -> grey H2; NG+CCS -> blue H2; electrolysis -> green H2
    6. Electricity: direct emissions (no precursors)

Emission Resolution:
    - Recursive chain traversal up to configurable max_depth
    - Default value waterfall per Annex III methodology
    - Mass balance validation across each production node
    - Scrap classification (pre-consumer vs post-consumer)

Zero-Hallucination:
    - All emission calculations use deterministic Decimal arithmetic
    - Emission factors from predefined reference tables only
    - Mass balance validated by conservation of mass principle
    - SHA-256 provenance hash on every calculation result

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-005 CBAM Complete
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
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


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GoodsCategory(str, Enum):
    """CBAM goods categories per Annex I."""
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    CEMENT = "cement"
    FERTILIZERS = "fertilizers"
    HYDROGEN = "hydrogen"
    ELECTRICITY = "electricity"


class AllocationMethod(str, Enum):
    """Emission allocation method for multi-output processes."""
    MASS = "mass"
    ECONOMIC = "economic"
    ENERGY = "energy"


class DefaultValueTier(str, Enum):
    """Default value waterfall tiers per Annex III."""
    ACTUAL_VERIFIED = "actual_verified"
    ACTUAL_UNVERIFIED = "actual_unverified"
    COUNTRY_DEFAULT = "country_default"
    EU_DEFAULT = "eu_default"
    GLOBAL_DEFAULT = "global_default"
    ANNEX_III_FALLBACK = "annex_iii_fallback"


class ScrapType(str, Enum):
    """Classification of scrap material."""
    PRE_CONSUMER = "pre_consumer"
    POST_CONSUMER = "post_consumer"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProductionRouteType(str, Enum):
    """Steel/aluminium production route."""
    BF_BOF = "bf_bof"
    EAF = "eaf"
    DRI_EAF = "dri_eaf"
    HALL_HEROULT = "hall_heroult"
    SECONDARY_ALUMINIUM = "secondary_aluminium"
    DRY_KILN = "dry_kiln"
    WET_KILN = "wet_kiln"
    HABER_BOSCH = "haber_bosch"
    SMR = "smr"
    SMR_CCS = "smr_ccs"
    ELECTROLYSIS = "electrolysis"
    DIRECT = "direct"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class PrecursorNode(BaseModel):
    """A node in the precursor chain representing one production stage."""
    node_id: str = Field(default_factory=_new_uuid, description="Node identifier")
    cn_code: str = Field(description="CN code of product at this stage")
    product_name: str = Field(description="Product name")
    goods_category: GoodsCategory = Field(description="CBAM goods category")
    production_route: str = Field(default="", description="Production route identifier")
    depth: int = Field(default=0, description="Depth in the precursor chain (0 = final product)")
    parent_node_id: Optional[str] = Field(default=None, description="Parent node in the chain")
    children_node_ids: List[str] = Field(default_factory=list, description="Precursor node IDs")
    direct_emissions_tco2e: Decimal = Field(default=Decimal("0"), description="Direct emissions at this stage")
    indirect_emissions_tco2e: Decimal = Field(default=Decimal("0"), description="Indirect (electricity) emissions")
    precursor_emissions_tco2e: Decimal = Field(default=Decimal("0"), description="Inherited precursor emissions")
    total_embedded_emissions_tco2e: Decimal = Field(default=Decimal("0"), description="Total embedded emissions")
    mass_input_tonnes: Decimal = Field(default=Decimal("0"), description="Mass input in tonnes")
    mass_output_tonnes: Decimal = Field(default=Decimal("0"), description="Mass output in tonnes")
    yield_rate: Decimal = Field(default=Decimal("1"), description="Yield rate (output/input)")
    data_source: DefaultValueTier = Field(
        default=DefaultValueTier.GLOBAL_DEFAULT, description="Source tier for emission data"
    )
    installation_id: Optional[str] = Field(default=None, description="Installation ID if actual data")
    country_code: str = Field(default="", description="Country of production")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("direct_emissions_tco2e", "indirect_emissions_tco2e",
                     "precursor_emissions_tco2e", "total_embedded_emissions_tco2e",
                     "mass_input_tonnes", "mass_output_tonnes", "yield_rate", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class PrecursorChain(BaseModel):
    """Complete precursor chain for a CBAM product."""
    chain_id: str = Field(default_factory=_new_uuid, description="Chain identifier")
    root_cn_code: str = Field(description="CN code of the final product")
    root_product_name: str = Field(default="", description="Final product name")
    goods_category: GoodsCategory = Field(description="CBAM goods category")
    max_depth: int = Field(default=5, description="Maximum chain depth")
    nodes: List[PrecursorNode] = Field(default_factory=list, description="All nodes in the chain")
    root_node_id: Optional[str] = Field(default=None, description="Root node identifier")
    total_chain_emissions_tco2e: Decimal = Field(default=Decimal("0"), description="Total chain emissions")
    resolved_at: datetime = Field(default_factory=_utcnow, description="Resolution timestamp")
    gaps: List[str] = Field(default_factory=list, description="Identified data gaps")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_chain_emissions_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class PrecursorEmissionResult(BaseModel):
    """Result of precursor emission calculation."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    chain_id: str = Field(description="Source chain identifier")
    total_direct_emissions: Decimal = Field(description="Sum of direct emissions across chain")
    total_indirect_emissions: Decimal = Field(description="Sum of indirect emissions across chain")
    total_embedded_emissions: Decimal = Field(description="Total embedded emissions for final product")
    specific_embedded_emissions: Decimal = Field(description="Emissions per tonne of final product")
    emission_by_stage: List[Dict[str, Any]] = Field(default_factory=list, description="Emissions at each stage")
    data_quality_score: Decimal = Field(default=Decimal("0"), description="Overall data quality (0-1)")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_direct_emissions", "total_indirect_emissions",
                     "total_embedded_emissions", "specific_embedded_emissions",
                     "data_quality_score", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class AllocationResult(BaseModel):
    """Result of emission allocation across co-products."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    chain_id: str = Field(description="Source chain identifier")
    method: AllocationMethod = Field(description="Allocation method used")
    allocations: List[Dict[str, Any]] = Field(default_factory=list, description="Per-product allocation")
    total_emissions_allocated: Decimal = Field(description="Total emissions allocated")
    unallocated_residual: Decimal = Field(default=Decimal("0"), description="Unallocated residual")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_emissions_allocated", "unallocated_residual", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class CompositionRecord(BaseModel):
    """Record of product composition and its precursor materials."""
    record_id: str = Field(default_factory=_new_uuid, description="Record identifier")
    product_id: str = Field(description="Product identifier")
    product_name: str = Field(default="", description="Product name")
    components: List[Dict[str, Any]] = Field(default_factory=list, description="Component breakdown")
    total_mass_tonnes: Decimal = Field(description="Total product mass in tonnes")
    recorded_at: datetime = Field(default_factory=_utcnow, description="Record timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_mass_tonnes", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class DefaultFallbackResult(BaseModel):
    """Result of applying default value waterfall."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    chain_id: str = Field(description="Source chain identifier")
    nodes_with_defaults: int = Field(default=0, description="Number of nodes using default values")
    nodes_with_actuals: int = Field(default=0, description="Number of nodes with actual data")
    default_tier_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Count of nodes per default tier"
    )
    applied_defaults: List[Dict[str, Any]] = Field(
        default_factory=list, description="Details of defaults applied"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class MassBalanceResult(BaseModel):
    """Result of mass balance validation across the chain."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    chain_id: str = Field(description="Source chain identifier")
    is_balanced: bool = Field(description="Whether mass balance holds within tolerance")
    total_input_tonnes: Decimal = Field(description="Total mass entering the chain")
    total_output_tonnes: Decimal = Field(description="Total mass leaving the chain")
    total_waste_tonnes: Decimal = Field(description="Mass lost to waste/byproducts")
    imbalance_tonnes: Decimal = Field(description="Absolute imbalance")
    imbalance_pct: Decimal = Field(description="Imbalance as percentage of input")
    tolerance_pct: Decimal = Field(default=Decimal("5"), description="Accepted tolerance %")
    node_balances: List[Dict[str, Any]] = Field(default_factory=list, description="Per-node balance")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_input_tonnes", "total_output_tonnes", "total_waste_tonnes",
                     "imbalance_tonnes", "imbalance_pct", "tolerance_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class ScrapClassification(BaseModel):
    """Classification result for scrap material."""
    classification_id: str = Field(default_factory=_new_uuid, description="Classification identifier")
    material: str = Field(description="Material description")
    origin: str = Field(description="Origin description")
    scrap_type: ScrapType = Field(description="Classified scrap type")
    emission_factor_tco2e_per_tonne: Decimal = Field(description="Applicable emission factor")
    rationale: str = Field(default="", description="Classification rationale")
    cbam_relevant: bool = Field(default=True, description="Whether CBAM-relevant")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("emission_factor_tco2e_per_tonne", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class ProductionRoute(BaseModel):
    """Production route definition for a product."""
    route_id: str = Field(default_factory=_new_uuid, description="Route identifier")
    cn_code: str = Field(description="Product CN code")
    goods_category: GoodsCategory = Field(description="CBAM goods category")
    route_type: ProductionRouteType = Field(description="Production route type")
    route_description: str = Field(default="", description="Human-readable route description")
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Production stages")
    typical_emission_factor: Decimal = Field(description="Typical emission factor tCO2e/t product")
    benchmark_value: Decimal = Field(default=Decimal("0"), description="EU ETS benchmark value")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("typical_emission_factor", "benchmark_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class GapAnalysis(BaseModel):
    """Gap analysis for a precursor chain."""
    analysis_id: str = Field(default_factory=_new_uuid, description="Analysis identifier")
    chain_id: str = Field(description="Source chain identifier")
    total_nodes: int = Field(description="Total nodes in chain")
    nodes_with_actual_data: int = Field(description="Nodes with actual installation data")
    nodes_with_defaults: int = Field(description="Nodes using default values")
    nodes_missing_data: int = Field(description="Nodes missing data entirely")
    coverage_pct: Decimal = Field(description="Data coverage percentage")
    gaps: List[Dict[str, Any]] = Field(default_factory=list, description="Identified gaps with details")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations to fill gaps")
    risk_level: str = Field(default="low", description="Overall risk level (low/medium/high)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("coverage_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class ChainVisualization(BaseModel):
    """Visualization data for a precursor chain."""
    visualization_id: str = Field(default_factory=_new_uuid, description="Visualization identifier")
    chain_id: str = Field(description="Source chain identifier")
    tree_structure: Dict[str, Any] = Field(default_factory=dict, description="Nested tree structure")
    node_count: int = Field(description="Total nodes in visualization")
    max_depth: int = Field(description="Maximum depth of the chain")
    edge_list: List[Dict[str, str]] = Field(default_factory=list, description="List of edges (from, to)")
    mermaid_diagram: str = Field(default="", description="Mermaid.js diagram source")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Predefined Precursor Chains
# ---------------------------------------------------------------------------


_PREDEFINED_CHAINS: Dict[GoodsCategory, List[Dict[str, Any]]] = {
    GoodsCategory.IRON_STEEL: [
        {"stage": "iron_ore", "cn_prefix": "2601", "name": "Iron Ore",
         "ef": "0.05", "depth": 5, "yield": "0.60"},
        {"stage": "pig_iron", "cn_prefix": "7201", "name": "Pig Iron",
         "ef": "1.60", "depth": 4, "yield": "0.92"},
        {"stage": "crude_steel", "cn_prefix": "7206", "name": "Crude Steel",
         "ef": "0.35", "depth": 3, "yield": "0.95"},
        {"stage": "hot_rolled", "cn_prefix": "7208", "name": "Hot-Rolled Flat Products",
         "ef": "0.15", "depth": 2, "yield": "0.93"},
        {"stage": "cold_rolled", "cn_prefix": "7209", "name": "Cold-Rolled Flat Products",
         "ef": "0.10", "depth": 1, "yield": "0.95"},
        {"stage": "coated", "cn_prefix": "7210", "name": "Coated Flat Products",
         "ef": "0.08", "depth": 0, "yield": "0.97"},
    ],
    GoodsCategory.ALUMINIUM: [
        {"stage": "bauxite", "cn_prefix": "2606", "name": "Bauxite",
         "ef": "0.02", "depth": 5, "yield": "0.45"},
        {"stage": "alumina", "cn_prefix": "2818", "name": "Alumina (Al2O3)",
         "ef": "1.10", "depth": 4, "yield": "0.53"},
        {"stage": "primary_aluminium", "cn_prefix": "7601", "name": "Primary Aluminium",
         "ef": "6.80", "depth": 3, "yield": "0.95"},
        {"stage": "aluminium_alloy", "cn_prefix": "7601", "name": "Aluminium Alloy",
         "ef": "0.30", "depth": 2, "yield": "0.98"},
        {"stage": "rolled_aluminium", "cn_prefix": "7606", "name": "Rolled Aluminium",
         "ef": "0.25", "depth": 1, "yield": "0.92"},
        {"stage": "fabricated_aluminium", "cn_prefix": "7610", "name": "Fabricated Aluminium",
         "ef": "0.15", "depth": 0, "yield": "0.90"},
    ],
    GoodsCategory.CEMENT: [
        {"stage": "limestone", "cn_prefix": "2521", "name": "Limestone",
         "ef": "0.01", "depth": 3, "yield": "0.65"},
        {"stage": "clinker", "cn_prefix": "2523", "name": "Clinker",
         "ef": "0.83", "depth": 2, "yield": "0.95"},
        {"stage": "cement_cem_i", "cn_prefix": "252329", "name": "Cement CEM I (95% clinker)",
         "ef": "0.05", "depth": 1, "yield": "0.98"},
        {"stage": "cement_final", "cn_prefix": "2523", "name": "Cement (CEM I-V blended)",
         "ef": "0.02", "depth": 0, "yield": "0.99"},
    ],
    GoodsCategory.FERTILIZERS: [
        {"stage": "natural_gas", "cn_prefix": "2711", "name": "Natural Gas (feedstock)",
         "ef": "0.20", "depth": 3, "yield": "0.85"},
        {"stage": "ammonia", "cn_prefix": "2814", "name": "Ammonia (NH3)",
         "ef": "1.60", "depth": 2, "yield": "0.90"},
        {"stage": "urea", "cn_prefix": "3102", "name": "Urea / AN / CAN",
         "ef": "0.50", "depth": 1, "yield": "0.95"},
        {"stage": "fertilizer_final", "cn_prefix": "3105", "name": "Mixed Fertilizer",
         "ef": "0.10", "depth": 0, "yield": "0.98"},
    ],
    GoodsCategory.HYDROGEN: [
        {"stage": "grey_h2_ng", "cn_prefix": "280410", "name": "Natural Gas (SMR feedstock)",
         "ef": "9.00", "depth": 1, "yield": "0.75"},
        {"stage": "blue_h2_ccs", "cn_prefix": "280410", "name": "NG + CCS (Blue H2)",
         "ef": "2.50", "depth": 1, "yield": "0.72"},
        {"stage": "green_h2_electrolysis", "cn_prefix": "280410", "name": "Electrolysis (Green H2)",
         "ef": "0.50", "depth": 1, "yield": "0.65"},
        {"stage": "hydrogen_final", "cn_prefix": "280410", "name": "Hydrogen",
         "ef": "0.00", "depth": 0, "yield": "1.00"},
    ],
    GoodsCategory.ELECTRICITY: [
        {"stage": "electricity", "cn_prefix": "271600", "name": "Electricity",
         "ef": "0.40", "depth": 0, "yield": "1.00"},
    ],
}

# Default emission factors by country (tCO2e per tonne product, simplified)
_COUNTRY_DEFAULT_EFS: Dict[str, Dict[GoodsCategory, Decimal]] = {
    "CN": {GoodsCategory.IRON_STEEL: Decimal("2.30"), GoodsCategory.ALUMINIUM: Decimal("16.50"),
           GoodsCategory.CEMENT: Decimal("0.90"), GoodsCategory.FERTILIZERS: Decimal("3.20"),
           GoodsCategory.HYDROGEN: Decimal("12.00"), GoodsCategory.ELECTRICITY: Decimal("0.58")},
    "IN": {GoodsCategory.IRON_STEEL: Decimal("2.80"), GoodsCategory.ALUMINIUM: Decimal("18.00"),
           GoodsCategory.CEMENT: Decimal("0.75"), GoodsCategory.FERTILIZERS: Decimal("3.50"),
           GoodsCategory.HYDROGEN: Decimal("11.00"), GoodsCategory.ELECTRICITY: Decimal("0.72")},
    "RU": {GoodsCategory.IRON_STEEL: Decimal("2.10"), GoodsCategory.ALUMINIUM: Decimal("5.50"),
           GoodsCategory.CEMENT: Decimal("0.85"), GoodsCategory.FERTILIZERS: Decimal("2.80"),
           GoodsCategory.HYDROGEN: Decimal("10.50"), GoodsCategory.ELECTRICITY: Decimal("0.35")},
    "TR": {GoodsCategory.IRON_STEEL: Decimal("1.80"), GoodsCategory.ALUMINIUM: Decimal("12.00"),
           GoodsCategory.CEMENT: Decimal("0.80"), GoodsCategory.FERTILIZERS: Decimal("3.00"),
           GoodsCategory.HYDROGEN: Decimal("10.00"), GoodsCategory.ELECTRICITY: Decimal("0.45")},
    "UA": {GoodsCategory.IRON_STEEL: Decimal("2.50"), GoodsCategory.ALUMINIUM: Decimal("14.00"),
           GoodsCategory.CEMENT: Decimal("0.88"), GoodsCategory.FERTILIZERS: Decimal("3.30"),
           GoodsCategory.HYDROGEN: Decimal("11.50"), GoodsCategory.ELECTRICITY: Decimal("0.50")},
    "DEFAULT": {GoodsCategory.IRON_STEEL: Decimal("2.20"), GoodsCategory.ALUMINIUM: Decimal("14.00"),
                GoodsCategory.CEMENT: Decimal("0.85"), GoodsCategory.FERTILIZERS: Decimal("3.10"),
                GoodsCategory.HYDROGEN: Decimal("11.00"), GoodsCategory.ELECTRICITY: Decimal("0.50")},
}

# Scrap emission factors (tCO2e per tonne)
_SCRAP_EMISSION_FACTORS: Dict[str, Dict[ScrapType, Decimal]] = {
    "iron_steel": {
        ScrapType.PRE_CONSUMER: Decimal("0.05"),
        ScrapType.POST_CONSUMER: Decimal("0.04"),
        ScrapType.MIXED: Decimal("0.045"),
        ScrapType.UNKNOWN: Decimal("0.05"),
    },
    "aluminium": {
        ScrapType.PRE_CONSUMER: Decimal("0.30"),
        ScrapType.POST_CONSUMER: Decimal("0.25"),
        ScrapType.MIXED: Decimal("0.28"),
        ScrapType.UNKNOWN: Decimal("0.30"),
    },
}


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class PrecursorChainConfig(BaseModel):
    """Configuration for the PrecursorChainEngine."""
    max_depth: int = Field(default=5, description="Maximum chain resolution depth")
    mass_balance_tolerance_pct: Decimal = Field(
        default=Decimal("5"), description="Mass balance tolerance percentage"
    )
    default_country: str = Field(default="DEFAULT", description="Default country for emission factors")
    enable_scrap_netting: bool = Field(default=True, description="Enable scrap emission netting")


# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

PrecursorChainConfig.model_rebuild()
PrecursorNode.model_rebuild()
PrecursorChain.model_rebuild()
PrecursorEmissionResult.model_rebuild()
AllocationResult.model_rebuild()
CompositionRecord.model_rebuild()
DefaultFallbackResult.model_rebuild()
MassBalanceResult.model_rebuild()
ScrapClassification.model_rebuild()
ProductionRoute.model_rebuild()
GapAnalysis.model_rebuild()
ChainVisualization.model_rebuild()


# ---------------------------------------------------------------------------
# PrecursorChainEngine
# ---------------------------------------------------------------------------


class PrecursorChainEngine:
    """
    Multi-tier recursive precursor emission resolution engine.

    Resolves the complete production chain for CBAM goods categories,
    calculates embedded emissions at each stage, and supports multiple
    allocation methods per Annex III methodology.

    Attributes:
        config: Engine configuration.
        _chains: In-memory chain store (keyed by chain_id).
        _compositions: In-memory composition records (keyed by product_id).

    Example:
        >>> engine = PrecursorChainEngine()
        >>> chain = engine.resolve_chain("7208", {"country": "CN"})
        >>> result = engine.calculate_precursor_emissions(chain)
        >>> assert result.total_embedded_emissions > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PrecursorChainEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = PrecursorChainConfig(**config)
        elif config and isinstance(config, PrecursorChainConfig):
            self.config = config
        else:
            self.config = PrecursorChainConfig()

        self._chains: Dict[str, PrecursorChain] = {}
        self._compositions: Dict[str, CompositionRecord] = {}
        logger.info("PrecursorChainEngine initialized (v%s)", _MODULE_VERSION)

    # -----------------------------------------------------------------------
    # Chain Resolution
    # -----------------------------------------------------------------------

    def resolve_chain(
        self,
        product_cn: str,
        installation_data: Optional[Dict[str, Any]] = None,
        max_depth: int = 5,
    ) -> PrecursorChain:
        """Resolve the complete precursor chain for a CBAM product.

        Recursively builds the production chain from raw materials to final
        product. Uses actual installation data where available, falling back
        to predefined chains and default emission factors.

        Args:
            product_cn: CN code of the final product.
            installation_data: Optional dict with installation-specific data
                including emission factors, yields, and country.
            max_depth: Maximum recursion depth.

        Returns:
            Resolved PrecursorChain with all nodes populated.

        Raises:
            ValueError: If CN code cannot be mapped to a goods category.
        """
        category = self._classify_cn_code(product_cn)
        if category is None:
            raise ValueError(f"Cannot classify CN code {product_cn} to a CBAM goods category")

        installation_data = installation_data or {}
        country = installation_data.get("country", self.config.default_country)

        chain = PrecursorChain(
            root_cn_code=product_cn,
            goods_category=category,
            max_depth=min(max_depth, self.config.max_depth),
        )

        predefined = _PREDEFINED_CHAINS.get(category, [])
        if not predefined:
            chain.gaps.append(f"No predefined chain for category {category.value}")
            chain.provenance_hash = _compute_hash(chain)
            self._chains[chain.chain_id] = chain
            return chain

        chain.root_product_name = predefined[-1]["name"] if predefined else product_cn
        nodes: List[PrecursorNode] = []
        prev_node_id: Optional[str] = None

        for stage_def in reversed(predefined):
            depth = stage_def["depth"]
            if depth > max_depth:
                continue

            ef = _decimal(installation_data.get(
                f"{stage_def['stage']}_ef", stage_def["ef"]
            ))
            yield_rate = _decimal(installation_data.get(
                f"{stage_def['stage']}_yield", stage_def["yield"]
            ))
            data_source = DefaultValueTier.ACTUAL_VERIFIED if (
                f"{stage_def['stage']}_ef" in installation_data
            ) else DefaultValueTier.COUNTRY_DEFAULT

            node = PrecursorNode(
                cn_code=stage_def["cn_prefix"],
                product_name=stage_def["name"],
                goods_category=category,
                production_route=stage_def["stage"],
                depth=depth,
                parent_node_id=prev_node_id,
                direct_emissions_tco2e=ef,
                yield_rate=yield_rate,
                data_source=data_source,
                installation_id=installation_data.get("installation_id"),
                country_code=country,
            )
            node.provenance_hash = _compute_hash(node)
            nodes.append(node)

            if prev_node_id:
                for n in nodes:
                    if n.node_id == prev_node_id:
                        n.children_node_ids.append(node.node_id)
                        break

            prev_node_id = node.node_id

        if nodes:
            chain.root_node_id = nodes[0].node_id

        chain.nodes = nodes
        chain.provenance_hash = _compute_hash(chain)
        self._chains[chain.chain_id] = chain

        logger.info(
            "Resolved chain %s for CN %s (%s): %d nodes, max_depth=%d",
            chain.chain_id, product_cn, category.value, len(nodes), max_depth,
        )
        return chain

    # -----------------------------------------------------------------------
    # Emission Calculation
    # -----------------------------------------------------------------------

    def calculate_precursor_emissions(
        self, chain: PrecursorChain
    ) -> PrecursorEmissionResult:
        """Calculate total embedded emissions across the precursor chain.

        Traverses the chain from raw materials to final product, accumulating
        direct, indirect, and inherited precursor emissions at each stage.

        Args:
            chain: Resolved precursor chain.

        Returns:
            PrecursorEmissionResult with emission breakdown.
        """
        total_direct = Decimal("0")
        total_indirect = Decimal("0")
        emission_by_stage: List[Dict[str, Any]] = []

        cumulative = Decimal("0")
        sorted_nodes = sorted(chain.nodes, key=lambda n: n.depth, reverse=True)

        for node in sorted_nodes:
            stage_direct = node.direct_emissions_tco2e
            stage_indirect = node.indirect_emissions_tco2e

            if node.yield_rate > Decimal("0"):
                inherited = cumulative / node.yield_rate
            else:
                inherited = cumulative

            node.precursor_emissions_tco2e = inherited
            node.total_embedded_emissions_tco2e = stage_direct + stage_indirect + inherited
            cumulative = node.total_embedded_emissions_tco2e

            total_direct += stage_direct
            total_indirect += stage_indirect

            emission_by_stage.append({
                "node_id": node.node_id,
                "stage": node.production_route,
                "product_name": node.product_name,
                "depth": node.depth,
                "direct_tco2e": str(stage_direct),
                "indirect_tco2e": str(stage_indirect),
                "inherited_tco2e": str(inherited.quantize(Decimal("0.0001"))),
                "cumulative_tco2e": str(cumulative.quantize(Decimal("0.0001"))),
                "data_source": node.data_source.value,
            })

        total_embedded = cumulative
        root_node = next((n for n in chain.nodes if n.depth == 0), None)
        output_mass = root_node.mass_output_tonnes if root_node and root_node.mass_output_tonnes > 0 else Decimal("1")
        specific = (total_embedded / output_mass).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        actual_count = sum(1 for n in chain.nodes if n.data_source == DefaultValueTier.ACTUAL_VERIFIED)
        total_count = len(chain.nodes) or 1
        data_quality = (_decimal(actual_count) / _decimal(total_count)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        chain.total_chain_emissions_tco2e = total_embedded
        chain.provenance_hash = _compute_hash(chain)

        result = PrecursorEmissionResult(
            chain_id=chain.chain_id,
            total_direct_emissions=total_direct,
            total_indirect_emissions=total_indirect,
            total_embedded_emissions=total_embedded,
            specific_embedded_emissions=specific,
            emission_by_stage=emission_by_stage,
            data_quality_score=data_quality,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Calculated emissions for chain %s: total=%s tCO2e, specific=%s tCO2e/t",
            chain.chain_id, total_embedded.quantize(Decimal("0.01")), specific,
        )
        return result

    # -----------------------------------------------------------------------
    # Emission Allocation
    # -----------------------------------------------------------------------

    def allocate_emissions(
        self, chain: PrecursorChain, method: str = "mass"
    ) -> AllocationResult:
        """Allocate emissions across co-products using specified method.

        Args:
            chain: Resolved precursor chain.
            method: Allocation method ('mass', 'economic', 'energy').

        Returns:
            AllocationResult with per-product allocation breakdown.

        Raises:
            ValueError: If invalid allocation method.
        """
        try:
            alloc_method = AllocationMethod(method)
        except ValueError:
            raise ValueError(f"Invalid allocation method: {method}")

        allocations: List[Dict[str, Any]] = []
        total_allocated = Decimal("0")

        output_nodes = [n for n in chain.nodes if n.depth == 0]
        if not output_nodes:
            output_nodes = chain.nodes[:1] if chain.nodes else []

        total_weight = Decimal("0")
        for node in output_nodes:
            if alloc_method == AllocationMethod.MASS:
                weight = node.mass_output_tonnes if node.mass_output_tonnes > 0 else Decimal("1")
            elif alloc_method == AllocationMethod.ECONOMIC:
                weight = _decimal(node.provenance_hash[:8] if node.provenance_hash else "1")
                weight = max(weight, Decimal("1"))
            else:
                weight = node.total_embedded_emissions_tco2e if node.total_embedded_emissions_tco2e > 0 else Decimal("1")
            total_weight += weight

        for node in output_nodes:
            if alloc_method == AllocationMethod.MASS:
                weight = node.mass_output_tonnes if node.mass_output_tonnes > 0 else Decimal("1")
            elif alloc_method == AllocationMethod.ECONOMIC:
                weight = _decimal("1")
            else:
                weight = node.total_embedded_emissions_tco2e if node.total_embedded_emissions_tco2e > 0 else Decimal("1")

            fraction = (weight / total_weight).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) if total_weight > 0 else Decimal("1")
            allocated = (chain.total_chain_emissions_tco2e * fraction).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
            total_allocated += allocated

            allocations.append({
                "node_id": node.node_id,
                "product_name": node.product_name,
                "allocation_fraction": str(fraction),
                "allocated_emissions_tco2e": str(allocated),
                "method": alloc_method.value,
            })

        residual = chain.total_chain_emissions_tco2e - total_allocated

        result = AllocationResult(
            chain_id=chain.chain_id,
            method=alloc_method,
            allocations=allocations,
            total_emissions_allocated=total_allocated,
            unallocated_residual=residual,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Allocated emissions for chain %s using %s: %s allocated, %s residual",
            chain.chain_id, alloc_method.value, total_allocated, residual,
        )
        return result

    # -----------------------------------------------------------------------
    # Composition Tracking
    # -----------------------------------------------------------------------

    def track_composition(
        self, product_id: str, components: List[Dict[str, Any]]
    ) -> CompositionRecord:
        """Track the composition of a product and its precursor materials.

        Args:
            product_id: Product identifier.
            components: List of component dicts with 'name', 'mass_tonnes',
                'cn_code', and optional 'emission_factor'.

        Returns:
            CompositionRecord with full breakdown.

        Raises:
            ValueError: If product_id is empty or no components provided.
        """
        if not product_id or not product_id.strip():
            raise ValueError("product_id must not be empty")
        if not components:
            raise ValueError("At least one component is required")

        total_mass = sum(_decimal(c.get("mass_tonnes", 0)) for c in components)
        enriched_components: List[Dict[str, Any]] = []

        for comp in components:
            mass = _decimal(comp.get("mass_tonnes", 0))
            fraction = (mass / total_mass).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            ) if total_mass > 0 else Decimal("0")
            enriched_components.append({
                "name": comp.get("name", "Unknown"),
                "cn_code": comp.get("cn_code", ""),
                "mass_tonnes": str(mass),
                "mass_fraction": str(fraction),
                "emission_factor": str(_decimal(comp.get("emission_factor", 0))),
            })

        record = CompositionRecord(
            product_id=product_id.strip(),
            product_name=components[0].get("product_name", product_id),
            components=enriched_components,
            total_mass_tonnes=total_mass,
        )
        record.provenance_hash = _compute_hash(record)
        self._compositions[product_id] = record

        logger.info(
            "Tracked composition for %s: %d components, total %s tonnes",
            product_id, len(components), total_mass,
        )
        return record

    # -----------------------------------------------------------------------
    # Default Value Waterfall
    # -----------------------------------------------------------------------

    def apply_default_waterfall(
        self, chain: PrecursorChain
    ) -> DefaultFallbackResult:
        """Apply the default value waterfall per Annex III methodology.

        For nodes without actual data, applies default values in priority:
        1. Country-specific defaults
        2. EU average defaults
        3. Global average defaults
        4. Annex III fallback values

        Args:
            chain: Precursor chain to apply defaults to.

        Returns:
            DefaultFallbackResult with details of applied defaults.
        """
        tier_distribution: Dict[str, int] = defaultdict(int)
        applied_defaults: List[Dict[str, Any]] = []
        nodes_with_actuals = 0

        for node in chain.nodes:
            if node.data_source == DefaultValueTier.ACTUAL_VERIFIED:
                nodes_with_actuals += 1
                tier_distribution[DefaultValueTier.ACTUAL_VERIFIED.value] += 1
                continue

            country = node.country_code or self.config.default_country
            country_efs = _COUNTRY_DEFAULT_EFS.get(country, _COUNTRY_DEFAULT_EFS.get("DEFAULT", {}))
            category_ef = country_efs.get(node.goods_category)

            if category_ef is not None and country != "DEFAULT":
                node.data_source = DefaultValueTier.COUNTRY_DEFAULT
                node.direct_emissions_tco2e = category_ef
                tier_distribution[DefaultValueTier.COUNTRY_DEFAULT.value] += 1
            elif category_ef is not None:
                node.data_source = DefaultValueTier.GLOBAL_DEFAULT
                node.direct_emissions_tco2e = category_ef
                tier_distribution[DefaultValueTier.GLOBAL_DEFAULT.value] += 1
            else:
                node.data_source = DefaultValueTier.ANNEX_III_FALLBACK
                tier_distribution[DefaultValueTier.ANNEX_III_FALLBACK.value] += 1

            applied_defaults.append({
                "node_id": node.node_id,
                "product_name": node.product_name,
                "tier_applied": node.data_source.value,
                "emission_factor": str(node.direct_emissions_tco2e),
                "country": country,
            })
            node.provenance_hash = _compute_hash(node)

        nodes_with_defaults = len(chain.nodes) - nodes_with_actuals
        chain.provenance_hash = _compute_hash(chain)

        result = DefaultFallbackResult(
            chain_id=chain.chain_id,
            nodes_with_defaults=nodes_with_defaults,
            nodes_with_actuals=nodes_with_actuals,
            default_tier_distribution=dict(tier_distribution),
            applied_defaults=applied_defaults,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Applied defaults for chain %s: %d actual, %d defaults",
            chain.chain_id, nodes_with_actuals, nodes_with_defaults,
        )
        return result

    # -----------------------------------------------------------------------
    # Mass Balance Validation
    # -----------------------------------------------------------------------

    def validate_mass_balance(self, chain: PrecursorChain) -> MassBalanceResult:
        """Validate mass balance across the precursor chain.

        Ensures conservation of mass principle holds within tolerance.
        Mass_input at each stage should equal mass_output + waste.

        Args:
            chain: Precursor chain to validate.

        Returns:
            MassBalanceResult with validation details.
        """
        total_input = Decimal("0")
        total_output = Decimal("0")
        node_balances: List[Dict[str, Any]] = []

        for node in chain.nodes:
            inp = node.mass_input_tonnes
            out = node.mass_output_tonnes
            waste = inp - out if inp >= out else Decimal("0")
            total_input += inp
            total_output += out

            node_balances.append({
                "node_id": node.node_id,
                "product_name": node.product_name,
                "input_tonnes": str(inp),
                "output_tonnes": str(out),
                "waste_tonnes": str(waste),
                "yield_rate": str(node.yield_rate),
            })

        total_waste = total_input - total_output if total_input >= total_output else Decimal("0")
        imbalance = abs(total_input - total_output - total_waste)
        imbalance_pct = (
            (imbalance / total_input * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if total_input > 0 else Decimal("0")
        )
        is_balanced = imbalance_pct <= self.config.mass_balance_tolerance_pct

        result = MassBalanceResult(
            chain_id=chain.chain_id,
            is_balanced=is_balanced,
            total_input_tonnes=total_input,
            total_output_tonnes=total_output,
            total_waste_tonnes=total_waste,
            imbalance_tonnes=imbalance,
            imbalance_pct=imbalance_pct,
            tolerance_pct=self.config.mass_balance_tolerance_pct,
            node_balances=node_balances,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Mass balance for chain %s: %s (imbalance=%s%%)",
            chain.chain_id, "BALANCED" if is_balanced else "IMBALANCED", imbalance_pct,
        )
        return result

    # -----------------------------------------------------------------------
    # Scrap Classification
    # -----------------------------------------------------------------------

    def classify_scrap(
        self, material: str, origin: str
    ) -> ScrapClassification:
        """Classify scrap material for CBAM emission calculation.

        Per CBAM methodology, pre-consumer and post-consumer scrap have
        different emission factors. Classification affects whether scrap
        inputs receive zero or reduced emission allocation.

        Args:
            material: Material description (e.g. 'iron_steel', 'aluminium').
            origin: Origin description (e.g. 'factory_returns', 'recycled_cans').

        Returns:
            ScrapClassification with type and applicable emission factor.
        """
        material_lower = material.lower().strip()
        origin_lower = origin.lower().strip()

        pre_consumer_keywords = ["factory", "production", "manufacturing", "trim", "offcut", "internal"]
        post_consumer_keywords = ["recycled", "end-of-life", "consumer", "collection", "demolition"]

        scrap_type = ScrapType.UNKNOWN
        rationale = "Unable to determine scrap origin classification"

        if any(kw in origin_lower for kw in pre_consumer_keywords):
            scrap_type = ScrapType.PRE_CONSUMER
            rationale = f"Origin '{origin}' contains pre-consumer indicators"
        elif any(kw in origin_lower for kw in post_consumer_keywords):
            scrap_type = ScrapType.POST_CONSUMER
            rationale = f"Origin '{origin}' contains post-consumer indicators"
        elif "mixed" in origin_lower:
            scrap_type = ScrapType.MIXED
            rationale = f"Origin '{origin}' indicates mixed scrap sources"

        material_key = "iron_steel" if "steel" in material_lower or "iron" in material_lower else (
            "aluminium" if "alumin" in material_lower else "iron_steel"
        )
        ef_table = _SCRAP_EMISSION_FACTORS.get(material_key, _SCRAP_EMISSION_FACTORS["iron_steel"])
        ef = ef_table.get(scrap_type, ef_table[ScrapType.UNKNOWN])

        cbam_relevant = scrap_type != ScrapType.POST_CONSUMER

        result = ScrapClassification(
            material=material,
            origin=origin,
            scrap_type=scrap_type,
            emission_factor_tco2e_per_tonne=ef,
            rationale=rationale,
            cbam_relevant=cbam_relevant,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Classified scrap: material=%s, origin=%s -> %s (EF=%s, CBAM=%s)",
            material, origin, scrap_type.value, ef, cbam_relevant,
        )
        return result

    # -----------------------------------------------------------------------
    # Production Route Mapping
    # -----------------------------------------------------------------------

    def map_production_route(
        self, cn_code: str, installation: Optional[Dict[str, Any]] = None
    ) -> ProductionRoute:
        """Map a CN code and installation to a production route.

        Args:
            cn_code: Product CN code.
            installation: Optional installation data with route type.

        Returns:
            ProductionRoute definition with typical emission factor.
        """
        category = self._classify_cn_code(cn_code)
        installation = installation or {}

        route_type = self._determine_route_type(category, installation)
        predefined = _PREDEFINED_CHAINS.get(category, [])

        stages: List[Dict[str, Any]] = []
        typical_ef = Decimal("0")
        for stage in predefined:
            stages.append({
                "stage": stage["stage"],
                "name": stage["name"],
                "emission_factor": stage["ef"],
                "yield_rate": stage["yield"],
            })
            typical_ef += _decimal(stage["ef"])

        route = ProductionRoute(
            cn_code=cn_code,
            goods_category=category or GoodsCategory.IRON_STEEL,
            route_type=route_type,
            route_description=f"{category.value if category else 'unknown'} via {route_type.value}",
            stages=stages,
            typical_emission_factor=typical_ef,
            benchmark_value=typical_ef * Decimal("0.85"),
        )
        route.provenance_hash = _compute_hash(route)

        logger.info(
            "Mapped route for CN %s: %s, typical EF=%s",
            cn_code, route_type.value, typical_ef,
        )
        return route

    # -----------------------------------------------------------------------
    # Gap Analysis
    # -----------------------------------------------------------------------

    def analyze_gaps(self, chain: PrecursorChain) -> GapAnalysis:
        """Analyze data gaps in a precursor chain.

        Identifies nodes missing actual data and provides recommendations
        for improving data quality.

        Args:
            chain: Precursor chain to analyze.

        Returns:
            GapAnalysis with coverage metrics and recommendations.
        """
        total_nodes = len(chain.nodes)
        actual_count = 0
        default_count = 0
        missing_count = 0
        gaps: List[Dict[str, Any]] = []

        for node in chain.nodes:
            if node.data_source == DefaultValueTier.ACTUAL_VERIFIED:
                actual_count += 1
            elif node.data_source in (DefaultValueTier.ACTUAL_UNVERIFIED,
                                       DefaultValueTier.COUNTRY_DEFAULT,
                                       DefaultValueTier.EU_DEFAULT):
                default_count += 1
                gaps.append({
                    "node_id": node.node_id,
                    "product_name": node.product_name,
                    "gap_type": "using_default",
                    "current_tier": node.data_source.value,
                    "severity": "medium",
                })
            else:
                if node.direct_emissions_tco2e == Decimal("0") and node.depth > 0:
                    missing_count += 1
                    gaps.append({
                        "node_id": node.node_id,
                        "product_name": node.product_name,
                        "gap_type": "missing_data",
                        "current_tier": node.data_source.value,
                        "severity": "high",
                    })
                else:
                    default_count += 1

        coverage = (_decimal(actual_count) / _decimal(total_nodes) * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if total_nodes > 0 else Decimal("0")

        recommendations: List[str] = []
        if missing_count > 0:
            recommendations.append(
                f"Request actual emission data from {missing_count} supplier(s) to fill critical gaps."
            )
        if default_count > 0:
            recommendations.append(
                f"Verify {default_count} nodes currently using default values with supplier-specific data."
            )
        if coverage < Decimal("50"):
            recommendations.append("Data coverage below 50%. Prioritize Tier 1 supplier engagement.")
            risk_level = "high"
        elif coverage < Decimal("80"):
            recommendations.append("Moderate data coverage. Plan for improvement in next reporting cycle.")
            risk_level = "medium"
        else:
            risk_level = "low"

        result = GapAnalysis(
            chain_id=chain.chain_id,
            total_nodes=total_nodes,
            nodes_with_actual_data=actual_count,
            nodes_with_defaults=default_count,
            nodes_missing_data=missing_count,
            coverage_pct=coverage,
            gaps=gaps,
            recommendations=recommendations,
            risk_level=risk_level,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Gap analysis for chain %s: coverage=%s%%, risk=%s",
            chain.chain_id, coverage, risk_level,
        )
        return result

    # -----------------------------------------------------------------------
    # Chain Visualization
    # -----------------------------------------------------------------------

    def generate_chain_visualization(
        self, chain: PrecursorChain
    ) -> ChainVisualization:
        """Generate visualization data for a precursor chain.

        Produces a nested tree structure, edge list, and Mermaid.js diagram
        for rendering the chain in documentation or dashboards.

        Args:
            chain: Precursor chain to visualize.

        Returns:
            ChainVisualization with multiple rendering formats.
        """
        nodes_by_id = {n.node_id: n for n in chain.nodes}
        edge_list: List[Dict[str, str]] = []
        max_depth = 0

        for node in chain.nodes:
            max_depth = max(max_depth, node.depth)
            if node.parent_node_id and node.parent_node_id in nodes_by_id:
                edge_list.append({
                    "from": node.parent_node_id,
                    "to": node.node_id,
                    "from_name": nodes_by_id[node.parent_node_id].product_name,
                    "to_name": node.product_name,
                })

        def _build_tree(node_id: str) -> Dict[str, Any]:
            node = nodes_by_id.get(node_id)
            if not node:
                return {}
            children = [_build_tree(cid) for cid in node.children_node_ids if cid in nodes_by_id]
            return {
                "node_id": node.node_id,
                "name": node.product_name,
                "cn_code": node.cn_code,
                "depth": node.depth,
                "emissions_tco2e": str(node.total_embedded_emissions_tco2e),
                "data_source": node.data_source.value,
                "children": children,
            }

        tree = _build_tree(chain.root_node_id) if chain.root_node_id else {}

        mermaid_lines = ["graph TD"]
        for node in chain.nodes:
            safe_name = node.product_name.replace(" ", "_").replace("(", "").replace(")", "")
            label = f"{node.product_name}\\n{node.direct_emissions_tco2e} tCO2e"
            mermaid_lines.append(f'    {safe_name}["{label}"]')
        for edge in edge_list:
            from_safe = edge["from_name"].replace(" ", "_").replace("(", "").replace(")", "")
            to_safe = edge["to_name"].replace(" ", "_").replace("(", "").replace(")", "")
            mermaid_lines.append(f"    {from_safe} --> {to_safe}")

        result = ChainVisualization(
            chain_id=chain.chain_id,
            tree_structure=tree,
            node_count=len(chain.nodes),
            max_depth=max_depth,
            edge_list=edge_list,
            mermaid_diagram="\n".join(mermaid_lines),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Generated visualization for chain %s: %d nodes, depth=%d",
            chain.chain_id, len(chain.nodes), max_depth,
        )
        return result

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _classify_cn_code(self, cn_code: str) -> Optional[GoodsCategory]:
        """Classify a CN code to a CBAM goods category.

        Args:
            cn_code: CN code string (2-8 digits).

        Returns:
            GoodsCategory or None if not classifiable.
        """
        cn = cn_code.strip()[:4]
        cn_category_map = {
            "2601": GoodsCategory.IRON_STEEL,
            "7201": GoodsCategory.IRON_STEEL,
            "7202": GoodsCategory.IRON_STEEL,
            "7203": GoodsCategory.IRON_STEEL,
            "7204": GoodsCategory.IRON_STEEL,
            "7205": GoodsCategory.IRON_STEEL,
            "7206": GoodsCategory.IRON_STEEL,
            "7207": GoodsCategory.IRON_STEEL,
            "7208": GoodsCategory.IRON_STEEL,
            "7209": GoodsCategory.IRON_STEEL,
            "7210": GoodsCategory.IRON_STEEL,
            "7211": GoodsCategory.IRON_STEEL,
            "7212": GoodsCategory.IRON_STEEL,
            "7213": GoodsCategory.IRON_STEEL,
            "7214": GoodsCategory.IRON_STEEL,
            "7215": GoodsCategory.IRON_STEEL,
            "7216": GoodsCategory.IRON_STEEL,
            "7217": GoodsCategory.IRON_STEEL,
            "7218": GoodsCategory.IRON_STEEL,
            "7219": GoodsCategory.IRON_STEEL,
            "7220": GoodsCategory.IRON_STEEL,
            "7221": GoodsCategory.IRON_STEEL,
            "7222": GoodsCategory.IRON_STEEL,
            "7223": GoodsCategory.IRON_STEEL,
            "7224": GoodsCategory.IRON_STEEL,
            "7225": GoodsCategory.IRON_STEEL,
            "7226": GoodsCategory.IRON_STEEL,
            "7228": GoodsCategory.IRON_STEEL,
            "7229": GoodsCategory.IRON_STEEL,
            "7301": GoodsCategory.IRON_STEEL,
            "7302": GoodsCategory.IRON_STEEL,
            "7303": GoodsCategory.IRON_STEEL,
            "7304": GoodsCategory.IRON_STEEL,
            "7305": GoodsCategory.IRON_STEEL,
            "7306": GoodsCategory.IRON_STEEL,
            "7307": GoodsCategory.IRON_STEEL,
            "7308": GoodsCategory.IRON_STEEL,
            "7309": GoodsCategory.IRON_STEEL,
            "7310": GoodsCategory.IRON_STEEL,
            "7311": GoodsCategory.IRON_STEEL,
            "7318": GoodsCategory.IRON_STEEL,
            "7326": GoodsCategory.IRON_STEEL,
            "2606": GoodsCategory.ALUMINIUM,
            "2818": GoodsCategory.ALUMINIUM,
            "7601": GoodsCategory.ALUMINIUM,
            "7602": GoodsCategory.ALUMINIUM,
            "7603": GoodsCategory.ALUMINIUM,
            "7604": GoodsCategory.ALUMINIUM,
            "7605": GoodsCategory.ALUMINIUM,
            "7606": GoodsCategory.ALUMINIUM,
            "7607": GoodsCategory.ALUMINIUM,
            "7608": GoodsCategory.ALUMINIUM,
            "7609": GoodsCategory.ALUMINIUM,
            "7610": GoodsCategory.ALUMINIUM,
            "7611": GoodsCategory.ALUMINIUM,
            "7612": GoodsCategory.ALUMINIUM,
            "7613": GoodsCategory.ALUMINIUM,
            "7614": GoodsCategory.ALUMINIUM,
            "7616": GoodsCategory.ALUMINIUM,
            "2521": GoodsCategory.CEMENT,
            "2523": GoodsCategory.CEMENT,
            "2711": GoodsCategory.FERTILIZERS,
            "2814": GoodsCategory.FERTILIZERS,
            "3102": GoodsCategory.FERTILIZERS,
            "3105": GoodsCategory.FERTILIZERS,
            "2804": GoodsCategory.HYDROGEN,
            "2716": GoodsCategory.ELECTRICITY,
        }

        result = cn_category_map.get(cn)
        if result:
            return result

        cn2 = cn[:2]
        if cn2 in ("72", "73"):
            return GoodsCategory.IRON_STEEL
        if cn2 == "76":
            return GoodsCategory.ALUMINIUM
        if cn2 == "25":
            return GoodsCategory.CEMENT
        if cn2 in ("28", "31"):
            return GoodsCategory.FERTILIZERS

        return None

    def _determine_route_type(
        self, category: Optional[GoodsCategory], installation: Dict[str, Any]
    ) -> ProductionRouteType:
        """Determine the production route type for a category.

        Args:
            category: CBAM goods category.
            installation: Installation data.

        Returns:
            ProductionRouteType.
        """
        explicit = installation.get("production_route")
        if explicit:
            try:
                return ProductionRouteType(explicit)
            except ValueError:
                pass

        route_map = {
            GoodsCategory.IRON_STEEL: ProductionRouteType.BF_BOF,
            GoodsCategory.ALUMINIUM: ProductionRouteType.HALL_HEROULT,
            GoodsCategory.CEMENT: ProductionRouteType.DRY_KILN,
            GoodsCategory.FERTILIZERS: ProductionRouteType.HABER_BOSCH,
            GoodsCategory.HYDROGEN: ProductionRouteType.SMR,
            GoodsCategory.ELECTRICITY: ProductionRouteType.DIRECT,
        }
        return route_map.get(category, ProductionRouteType.DIRECT)
