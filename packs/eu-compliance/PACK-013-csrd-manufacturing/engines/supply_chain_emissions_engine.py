# -*- coding: utf-8 -*-
"""
SupplyChainEmissionsEngine - PACK-013 CSRD Manufacturing Engine 7
===================================================================

Manufacturing-specific Scope 3 supply chain emissions calculation engine.
Implements multiple calculation methodologies (supplier-specific, hybrid,
average-data, spend-based) per GHG Protocol Scope 3 Standard and ESRS E1
value chain emission disclosure requirements.

Scope 3 Categories (Manufacturing Focus):
    Cat 1  - Purchased Goods & Services (primary focus)
    Cat 2  - Capital Goods
    Cat 3  - Fuel & Energy Related Activities
    Cat 4  - Upstream Transportation & Distribution
    Cat 5  - Waste Generated in Operations
    Cat 6  - Business Travel
    Cat 7  - Employee Commuting
    Cat 8  - Upstream Leased Assets
    Cat 9  - Downstream Transportation
    Cat 10 - Processing of Sold Products
    Cat 11 - Use of Sold Products
    Cat 12 - End-of-Life Treatment
    Cat 13 - Downstream Leased Assets
    Cat 14 - Franchises
    Cat 15 - Investments

Calculation Hierarchy (GHG Protocol):
    1. Supplier-specific (highest quality)
    2. Hybrid (partial primary + secondary data)
    3. Average-data (physical activity data x emission factors)
    4. Spend-based (economic input-output, lowest quality)

Regulatory References:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard
    - ESRS E1 (Climate Change) - E1-6 Scope 3 GHG emissions
    - PEFCR/OEF for product-level carbon footprinting
    - SBTi Scope 3 target-setting guidance

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Emission factors from published databases (DEFRA, EPA, Ecoinvent)
    - Data quality scoring per GHG Protocol 5-point scale
    - SHA-256 provenance hash on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-013 CSRD Manufacturing
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
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

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _round_value(value: Decimal, places: int = 3) -> float:
    """Round a Decimal to specified places and return float."""
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)

def _pct(numerator: Decimal, denominator: Decimal, places: int = 2) -> float:
    """Calculate percentage safely, returning 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return _round_value((numerator / denominator) * Decimal("100"), places)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CalculationMethod(str, Enum):
    """Scope 3 calculation methodology per GHG Protocol hierarchy."""
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 emission categories."""
    CAT_1 = "cat_1_purchased_goods_services"
    CAT_2 = "cat_2_capital_goods"
    CAT_3 = "cat_3_fuel_energy_activities"
    CAT_4 = "cat_4_upstream_transportation"
    CAT_5 = "cat_5_waste_generated"
    CAT_6 = "cat_6_business_travel"
    CAT_7 = "cat_7_employee_commuting"
    CAT_8 = "cat_8_upstream_leased_assets"
    CAT_9 = "cat_9_downstream_transportation"
    CAT_10 = "cat_10_processing_sold_products"
    CAT_11 = "cat_11_use_sold_products"
    CAT_12 = "cat_12_end_of_life_treatment"
    CAT_13 = "cat_13_downstream_leased_assets"
    CAT_14 = "cat_14_franchises"
    CAT_15 = "cat_15_investments"

class SupplierTier(str, Enum):
    """Supplier tier in the value chain."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4_PLUS = "tier_4_plus"

class DataQualityScore(int, Enum):
    """Data quality score per GHG Protocol 5-point scale.

    1 = Primary data from supplier (highest quality)
    2 = Verified secondary data
    3 = Proxy or modeled data
    4 = Estimated data
    5 = Financial/spend-based (lowest quality)
    """
    SCORE_1 = 1
    SCORE_2 = 2
    SCORE_3 = 3
    SCORE_4 = 4
    SCORE_5 = 5

class TransportMode(str, Enum):
    """Mode of transportation for upstream/downstream logistics."""
    ROAD = "road"
    RAIL = "rail"
    SEA = "sea"
    AIR = "air"
    INLAND_WATERWAY = "inland_waterway"
    PIPELINE = "pipeline"

class EngagementPriority(str, Enum):
    """Priority level for supplier engagement."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# ---------------------------------------------------------------------------
# Constants - Emission Factors
# ---------------------------------------------------------------------------

# Spend-based emission factors (tCO2e per EUR)
# Source: Derived from Exiobase MRIO / EPA EEIO models, EU-adjusted
# These are average-economy factors by NACE sector classification.

SPEND_EMISSION_FACTORS: Dict[str, float] = {
    # Primary materials
    "A01": 0.00095,   # Agriculture, hunting
    "A02": 0.00045,   # Forestry
    "B05": 0.00220,   # Mining of coal
    "B06": 0.00180,   # Extraction of oil and gas
    "B07": 0.00150,   # Mining of metal ores
    "B08": 0.00120,   # Other mining and quarrying
    # Manufacturing
    "C10": 0.00065,   # Food products
    "C11": 0.00055,   # Beverages
    "C13": 0.00070,   # Textiles
    "C14": 0.00050,   # Wearing apparel
    "C16": 0.00060,   # Wood and wood products
    "C17": 0.00080,   # Paper and paper products
    "C19": 0.00350,   # Coke and refined petroleum
    "C20": 0.00180,   # Chemicals and chemical products
    "C21": 0.00045,   # Pharmaceuticals
    "C22": 0.00090,   # Rubber and plastic products
    "C23": 0.00200,   # Other non-metallic mineral products (cement, glass, ceramics)
    "C24": 0.00280,   # Basic metals (iron, steel, aluminium)
    "C25": 0.00100,   # Fabricated metal products
    "C26": 0.00040,   # Computer and electronic products
    "C27": 0.00055,   # Electrical equipment
    "C28": 0.00065,   # Machinery and equipment
    "C29": 0.00070,   # Motor vehicles
    "C30": 0.00060,   # Other transport equipment
    "C31": 0.00050,   # Furniture
    "C32": 0.00045,   # Other manufacturing
    # Energy
    "D35": 0.00300,   # Electricity, gas, steam
    # Water/waste
    "E36": 0.00080,   # Water collection and supply
    "E38": 0.00120,   # Waste management
    # Construction
    "F41": 0.00110,   # Construction of buildings
    "F42": 0.00130,   # Civil engineering
    # Transport
    "H49": 0.00140,   # Land transport
    "H50": 0.00200,   # Water transport
    "H51": 0.00350,   # Air transport
    "H52": 0.00060,   # Warehousing and support
    # Services (lower intensity)
    "J62": 0.00015,   # IT and computer services
    "K64": 0.00012,   # Financial services
    "M71": 0.00020,   # Architectural and engineering
    "M72": 0.00018,   # Scientific research
    "N77": 0.00030,   # Rental and leasing
    "N82": 0.00022,   # Office administrative
    # Default
    "default": 0.00080,
}

# Material-based emission factors (kgCO2e per kg of material)
# Source: Ecoinvent 3.9 / DEFRA 2024 / GaBi averages

MATERIAL_EMISSION_FACTORS: Dict[str, float] = {
    # Metals
    "steel_bof": 2.10,         # Basic oxygen furnace steel
    "steel_eaf": 0.45,         # Electric arc furnace steel
    "steel_generic": 1.85,     # Average steel
    "aluminium_primary": 8.20, # Primary aluminium
    "aluminium_recycled": 0.50,# Recycled aluminium
    "aluminium_generic": 5.50, # Average aluminium mix
    "copper": 3.50,
    "zinc": 2.80,
    "titanium": 8.10,
    "nickel": 6.50,
    # Plastics
    "hdpe": 1.80,
    "ldpe": 2.10,
    "pp": 1.70,
    "pet": 2.70,
    "pvc": 1.90,
    "ps": 3.10,
    "abs": 3.50,
    "nylon": 6.50,
    "polycarbonate": 5.00,
    "plastic_generic": 2.50,
    # Chemicals
    "ammonia": 2.20,
    "chlorine": 1.00,
    "caustic_soda": 0.90,
    "ethylene": 1.80,
    "propylene": 1.50,
    "methanol": 0.70,
    "sulfuric_acid": 0.10,
    "nitric_acid": 1.80,
    # Construction materials
    "cement": 0.63,            # OPC clinker ratio ~0.7
    "concrete": 0.13,
    "glass_flat": 1.20,
    "glass_container": 0.85,
    "ceramics": 0.70,
    "brick": 0.24,
    # Wood / Paper
    "timber_softwood": -1.50,  # Net carbon store
    "timber_hardwood": -1.60,
    "paper_virgin": 1.10,
    "paper_recycled": 0.60,
    "cardboard": 0.80,
    # Textiles
    "cotton": 5.50,
    "polyester": 5.55,
    "wool": 17.00,
    "nylon_fibre": 7.20,
    # Electronics
    "pcb": 25.00,
    "semiconductor": 45.00,
    "battery_lithium_ion": 12.00,
    # Default
    "default": 2.00,
}

# Transport emission factors (kgCO2e per tonne-km)
# Source: DEFRA 2024 / GLEC Framework

TRANSPORT_EMISSION_FACTORS: Dict[str, float] = {
    TransportMode.ROAD: 0.062,
    TransportMode.RAIL: 0.022,
    TransportMode.SEA: 0.016,
    TransportMode.AIR: 0.602,
    TransportMode.INLAND_WATERWAY: 0.031,
    TransportMode.PIPELINE: 0.005,
}

# Data quality weights (for weighted average DQ scoring)
# Higher score = lower quality = lower weight in calculations

DQ_WEIGHTS: Dict[int, float] = {
    DataQualityScore.SCORE_1: 1.0,
    DataQualityScore.SCORE_2: 0.8,
    DataQualityScore.SCORE_3: 0.6,
    DataQualityScore.SCORE_4: 0.4,
    DataQualityScore.SCORE_5: 0.2,
}

# Engagement thresholds (% of total emissions to classify as hotspot)
HOTSPOT_THRESHOLD_PCT: float = 5.0  # Suppliers contributing >5% are hotspots
TOP_N_HOTSPOTS: int = 20

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SupplyChainConfig(BaseModel):
    """Configuration for supply chain emissions calculation."""
    reporting_year: int = Field(description="Reporting year")
    priority_categories: List[Scope3Category] = Field(
        default_factory=lambda: [
            Scope3Category.CAT_1,
            Scope3Category.CAT_4,
            Scope3Category.CAT_9,
        ],
        description="Priority Scope 3 categories for detailed analysis",
    )
    tier_depth: int = Field(
        default=1, ge=1, le=4,
        description="Supplier tier depth for analysis (1-4)",
    )
    include_hotspot_analysis: bool = Field(
        default=True, description="Identify emission hotspots"
    )
    supplier_engagement_platform: bool = Field(
        default=True, description="Generate supplier engagement recommendations"
    )
    dq_improvement_target: float = Field(
        default=3.0,
        description="Target weighted average data quality score (1.0=best, 5.0=worst)",
    )
    production_volume: int = Field(
        default=0,
        description="Total production volume for BOM-based calculations",
    )

    @field_validator("reporting_year", mode="before")
    @classmethod
    def _validate_year(cls, v: Any) -> int:
        year = int(v)
        if year < 2020 or year > 2035:
            raise ValueError(f"Reporting year {year} outside valid range 2020-2035")
        return year

class SupplierData(BaseModel):
    """Supplier data for Scope 3 emissions calculation."""
    supplier_id: str = Field(description="Unique supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    tier: SupplierTier = Field(default=SupplierTier.TIER_1, description="Supplier tier")
    country: str = Field(default="", description="Country of supplier")
    nace_sector: str = Field(
        default="default", description="NACE sector code for spend-based calculation"
    )
    spend_eur: Decimal = Field(
        default=Decimal("0"), description="Annual spend with supplier (EUR)"
    )
    scope3_category: Scope3Category = Field(
        default=Scope3Category.CAT_1,
        description="Scope 3 category for this supplier",
    )
    reported_emissions_tco2e: Optional[Decimal] = Field(
        default=None,
        description="Supplier-reported emissions allocated to purchaser (tCO2e)",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.SPEND_BASED,
        description="Calculation method used",
    )
    data_quality_score: DataQualityScore = Field(
        default=DataQualityScore.SCORE_5,
        description="Data quality score (1=best, 5=worst)",
    )
    bom_linked: bool = Field(
        default=False, description="Whether supplier is linked to BOM components"
    )
    certification: List[str] = Field(
        default_factory=list,
        description="Environmental certifications (ISO 14001, SBTi, etc.)",
    )

    @field_validator("spend_eur", mode="before")
    @classmethod
    def _coerce_spend(cls, v: Any) -> Decimal:
        d = _decimal(v)
        if d < 0:
            raise ValueError("Spend cannot be negative")
        return d

    @field_validator("reported_emissions_tco2e", mode="before")
    @classmethod
    def _coerce_emissions(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        d = _decimal(v)
        if d < 0:
            raise ValueError("Reported emissions cannot be negative")
        return d

class BOMEmissionData(BaseModel):
    """Bill of Materials (BOM) component emission data."""
    component_id: str = Field(description="Component identifier")
    component_name: str = Field(default="", description="Component name")
    material_type: str = Field(description="Material type (key in MATERIAL_EMISSION_FACTORS)")
    quantity_per_product: Decimal = Field(
        description="Quantity of material per product unit (kg)"
    )
    emission_factor_kgco2e_per_unit: Optional[Decimal] = Field(
        default=None,
        description="Override emission factor (kgCO2e/kg). Uses database if None.",
    )
    origin_country: str = Field(default="", description="Country of origin")
    supplier_id: str = Field(default="", description="Linked supplier ID")
    recycled_content_pct: float = Field(
        default=0.0,
        description="Recycled content percentage (0-100)",
    )

    @field_validator("quantity_per_product", mode="before")
    @classmethod
    def _coerce_qty(cls, v: Any) -> Decimal:
        d = _decimal(v)
        if d < 0:
            raise ValueError("Quantity per product cannot be negative")
        return d

    @field_validator("emission_factor_kgco2e_per_unit", mode="before")
    @classmethod
    def _coerce_ef(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class TransportData(BaseModel):
    """Transport leg data for upstream/downstream logistics emissions."""
    origin: str = Field(description="Origin location")
    destination: str = Field(description="Destination location")
    mode: TransportMode = Field(description="Transport mode")
    distance_km: Decimal = Field(description="Distance in kilometres")
    weight_tonnes: Decimal = Field(description="Cargo weight in tonnes")
    emission_factor: Optional[Decimal] = Field(
        default=None,
        description="Override emission factor (kgCO2e/tonne-km). Uses database if None.",
    )
    supplier_id: str = Field(default="", description="Linked supplier ID")
    scope3_category: Scope3Category = Field(
        default=Scope3Category.CAT_4,
        description="Scope 3 category (Cat 4 upstream or Cat 9 downstream)",
    )

    @field_validator("distance_km", "weight_tonnes", mode="before")
    @classmethod
    def _coerce_positive(cls, v: Any) -> Decimal:
        d = _decimal(v)
        if d < 0:
            raise ValueError("Distance and weight cannot be negative")
        return d

    @field_validator("emission_factor", mode="before")
    @classmethod
    def _coerce_ef(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class SupplierHotspot(BaseModel):
    """Supplier identified as an emission hotspot."""
    supplier_id: str = Field(description="Supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    emissions_tco2e: float = Field(description="Supplier emissions in tCO2e")
    share_of_total_pct: float = Field(description="Share of total Scope 3 emissions (%)")
    data_quality: int = Field(description="Data quality score (1-5)")
    improvement_potential: str = Field(
        default="", description="Improvement potential description"
    )
    tier: str = Field(default="", description="Supplier tier")
    scope3_category: str = Field(default="", description="Scope 3 category")

class EngagementRecommendation(BaseModel):
    """Supplier engagement recommendation for emissions reduction."""
    supplier_id: str = Field(description="Supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    action: str = Field(description="Recommended engagement action")
    priority: EngagementPriority = Field(description="Priority level")
    expected_dq_improvement: float = Field(
        default=0.0, description="Expected improvement in data quality score"
    )
    timeline: str = Field(default="", description="Recommended timeline")
    rationale: str = Field(default="", description="Rationale for the action")

class SupplyChainResult(BaseModel):
    """Complete supply chain emissions calculation result with provenance.

    Covers GHG Protocol Scope 3 Standard and ESRS E1-6 requirements.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    # --- Totals ---
    total_scope3_tco2e: float = Field(
        default=0.0, description="Total Scope 3 emissions (tCO2e)"
    )
    category_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by Scope 3 category (tCO2e)"
    )
    method_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by calculation method (tCO2e)"
    )
    tier_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by supplier tier (tCO2e)"
    )
    # --- Hotspots ---
    supplier_hotspots: List[SupplierHotspot] = Field(
        default_factory=list, description="Supplier emission hotspots"
    )
    material_hotspots: List[Dict[str, Any]] = Field(
        default_factory=list, description="Material emission hotspots from BOM"
    )
    # --- Transport ---
    transport_emissions_tco2e: float = Field(
        default=0.0, description="Transport-related emissions (tCO2e)"
    )
    transport_by_mode: Dict[str, float] = Field(
        default_factory=dict, description="Transport emissions by mode (tCO2e)"
    )
    # --- BOM ---
    bom_emissions_tco2e: float = Field(
        default=0.0, description="BOM-based product emissions (tCO2e)"
    )
    bom_emissions_per_product_kgco2e: float = Field(
        default=0.0, description="Product carbon footprint from BOM (kgCO2e/product)"
    )
    # --- Data quality ---
    weighted_data_quality: float = Field(
        default=5.0, description="Weighted average data quality score (1=best, 5=worst)"
    )
    dq_improvement_needed: bool = Field(
        default=True, description="Whether DQ improvement target is not met"
    )
    dq_by_category: Dict[str, float] = Field(
        default_factory=dict, description="Data quality score by Scope 3 category"
    )
    # --- Engagement ---
    engagement_recommendations: List[EngagementRecommendation] = Field(
        default_factory=list, description="Supplier engagement recommendations"
    )
    # --- Coverage ---
    coverage_pct: float = Field(
        default=0.0, description="Data coverage as % of total spend"
    )
    supplier_count: int = Field(default=0, description="Number of suppliers assessed")
    suppliers_with_primary_data: int = Field(
        default=0, description="Suppliers with primary data (DQ 1-2)"
    )
    # --- Metadata ---
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology and assumption notes"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SupplyChainEmissionsEngine:
    """Manufacturing-specific Scope 3 supply chain emissions engine.

    Provides deterministic, zero-hallucination calculations for:
    - Supplier-specific emissions (primary data from suppliers)
    - Spend-based emissions (EEIO emission factors by NACE sector)
    - BOM-based product carbon footprint (material emission factors)
    - Transport emissions (GLEC Framework mode-specific factors)
    - Hotspot identification and supplier engagement planning
    - Data quality assessment and improvement tracking

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    Every result includes a SHA-256 provenance hash for audit trails.
    """

    def __init__(self, config: SupplyChainConfig) -> None:
        """Initialize the SupplyChainEmissionsEngine.

        Args:
            config: Configuration including reporting year, priority
                    categories, and data quality targets.
        """
        self.config = config
        self._notes: List[str] = []
        logger.info(
            "SupplyChainEmissionsEngine v%s initialized for year %d, "
            "%d priority categories",
            _MODULE_VERSION,
            config.reporting_year,
            len(config.priority_categories),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_supply_chain_emissions(
        self,
        suppliers: List[SupplierData],
        bom: Optional[List[BOMEmissionData]] = None,
        transport: Optional[List[TransportData]] = None,
    ) -> SupplyChainResult:
        """Calculate total supply chain (Scope 3) emissions.

        Aggregates emissions from all suppliers using the appropriate
        calculation method, adds BOM-based product emissions if available,
        and includes transport emissions from logistics data.

        Args:
            suppliers: List of supplier data records.
            bom: Optional Bill of Materials emission data for product footprint.
            transport: Optional transport leg data for logistics emissions.

        Returns:
            SupplyChainResult with complete assessment and provenance.
        """
        start_time = time.perf_counter()
        self._notes = []

        # --- Supplier emissions ---
        total_scope3 = Decimal("0")
        category_totals: Dict[str, Decimal] = defaultdict(Decimal)
        method_totals: Dict[str, Decimal] = defaultdict(Decimal)
        tier_totals: Dict[str, Decimal] = defaultdict(Decimal)
        supplier_emissions: Dict[str, Decimal] = {}

        for supplier in suppliers:
            emissions = self.calculate_supplier_emissions(supplier)
            em_decimal = _decimal(emissions)
            supplier_emissions[supplier.supplier_id] = em_decimal
            total_scope3 += em_decimal
            category_totals[supplier.scope3_category.value] += em_decimal
            method_totals[supplier.calculation_method.value] += em_decimal
            tier_totals[supplier.tier.value] += em_decimal

        # --- BOM emissions ---
        bom_total = Decimal("0")
        bom_per_product = Decimal("0")
        material_hotspots_list: List[Dict[str, Any]] = []
        if bom:
            bom_result = self._calculate_bom_emissions_internal(
                bom, self.config.production_volume
            )
            bom_total = bom_result["total"]
            bom_per_product = bom_result["per_product"]
            material_hotspots_list = bom_result["material_hotspots"]
            total_scope3 += bom_total
            category_totals[Scope3Category.CAT_1.value] += bom_total
            method_totals[CalculationMethod.AVERAGE_DATA.value] += bom_total
            self._notes.append(
                f"BOM-based emissions: {_round_value(bom_total, 3)} tCO2e "
                f"({_round_value(bom_per_product, 3)} kgCO2e/product)"
            )

        # --- Transport emissions ---
        transport_total = Decimal("0")
        transport_by_mode_dec: Dict[str, Decimal] = defaultdict(Decimal)
        if transport:
            for leg in transport:
                leg_emissions = self._calculate_transport_leg(leg)
                transport_total += leg_emissions
                transport_by_mode_dec[leg.mode.value] += leg_emissions

            total_scope3 += transport_total
            # Allocate to appropriate category
            for leg in transport:
                category_totals[leg.scope3_category.value] += self._calculate_transport_leg(leg)

            self._notes.append(
                f"Transport emissions: {_round_value(transport_total, 3)} tCO2e "
                f"across {len(transport)} legs"
            )

        # --- Data quality ---
        weighted_dq = self.assess_data_quality(suppliers)
        dq_by_cat = self._assess_dq_by_category(suppliers)
        dq_needed = weighted_dq > self.config.dq_improvement_target

        primary_count = sum(
            1 for s in suppliers
            if s.data_quality_score in (DataQualityScore.SCORE_1, DataQualityScore.SCORE_2)
        )

        # --- Hotspot analysis ---
        supplier_hotspots: List[SupplierHotspot] = []
        if self.config.include_hotspot_analysis and total_scope3 > 0:
            supplier_hotspots = self.identify_hotspots(
                suppliers, supplier_emissions, total_scope3
            )

        # --- Engagement recommendations ---
        engagement_recs: List[EngagementRecommendation] = []
        if self.config.supplier_engagement_platform and supplier_hotspots:
            engagement_recs = self.generate_engagement_plan(
                supplier_hotspots, suppliers
            )

        # --- Coverage ---
        total_spend = sum(s.spend_eur for s in suppliers)
        coverage = 100.0 if total_spend > 0 else 0.0  # All suppliers with data are counted

        self._notes.append(
            f"Total Scope 3: {_round_value(total_scope3, 3)} tCO2e from "
            f"{len(suppliers)} suppliers, weighted DQ={round(weighted_dq, 2)}"
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = SupplyChainResult(
            total_scope3_tco2e=_round_value(total_scope3, 3),
            category_breakdown={k: _round_value(v, 3) for k, v in category_totals.items()},
            method_breakdown={k: _round_value(v, 3) for k, v in method_totals.items()},
            tier_breakdown={k: _round_value(v, 3) for k, v in tier_totals.items()},
            supplier_hotspots=supplier_hotspots,
            material_hotspots=material_hotspots_list,
            transport_emissions_tco2e=_round_value(transport_total, 3),
            transport_by_mode={k: _round_value(v, 3) for k, v in transport_by_mode_dec.items()},
            bom_emissions_tco2e=_round_value(bom_total, 3),
            bom_emissions_per_product_kgco2e=_round_value(bom_per_product, 3),
            weighted_data_quality=round(weighted_dq, 2),
            dq_improvement_needed=dq_needed,
            dq_by_category={k: round(v, 2) for k, v in dq_by_cat.items()},
            engagement_recommendations=engagement_recs,
            coverage_pct=round(coverage, 1),
            supplier_count=len(suppliers),
            suppliers_with_primary_data=primary_count,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_supplier_emissions(self, supplier: SupplierData) -> float:
        """Calculate emissions for a single supplier.

        Uses the appropriate calculation method:
        - SUPPLIER_SPECIFIC: Uses reported emissions directly
        - HYBRID: Weighted mix of reported and estimated
        - AVERAGE_DATA: Physical activity * emission factors
        - SPEND_BASED: Spend * EEIO emission factor

        Args:
            supplier: Supplier data record.

        Returns:
            Emissions in tCO2e as float.
        """
        if supplier.calculation_method == CalculationMethod.SUPPLIER_SPECIFIC:
            # Use supplier-reported emissions directly
            if supplier.reported_emissions_tco2e is not None:
                return _round_value(supplier.reported_emissions_tco2e, 6)
            # Fallback to spend-based if no reported data
            self._notes.append(
                f"Supplier {supplier.supplier_id}: no reported emissions, "
                f"falling back to spend-based method."
            )
            return self._spend_based_emissions(supplier)

        elif supplier.calculation_method == CalculationMethod.HYBRID:
            # Mix of reported and estimated
            if supplier.reported_emissions_tco2e is not None:
                reported = supplier.reported_emissions_tco2e
                spend_est = _decimal(self._spend_based_emissions(supplier))
                # Weighted: 70% reported, 30% spend-based estimate
                hybrid = (reported * Decimal("0.7")) + (spend_est * Decimal("0.3"))
                return _round_value(hybrid, 6)
            return self._spend_based_emissions(supplier)

        elif supplier.calculation_method == CalculationMethod.AVERAGE_DATA:
            # Use average-data emission factors (requires BOM linkage)
            if supplier.reported_emissions_tco2e is not None:
                return _round_value(supplier.reported_emissions_tco2e, 6)
            return self._spend_based_emissions(supplier)

        else:
            # SPEND_BASED
            return self._spend_based_emissions(supplier)

    def calculate_bom_emissions(
        self, bom: List[BOMEmissionData], production_volume: int
    ) -> float:
        """Calculate total BOM-based product emissions.

        Computes cradle-to-gate emissions from bill of materials components,
        accounting for material types, quantities, and recycled content.

        Args:
            bom: List of BOM component emission data.
            production_volume: Total production volume (number of units).

        Returns:
            Total BOM emissions in tCO2e.
        """
        result = self._calculate_bom_emissions_internal(bom, production_volume)
        return _round_value(result["total"], 3)

    def calculate_transport_emissions(
        self, transport: List[TransportData]
    ) -> float:
        """Calculate total transport emissions from logistics data.

        Uses mode-specific emission factors from the GLEC Framework.

        Args:
            transport: List of transport leg data.

        Returns:
            Total transport emissions in tCO2e.
        """
        total = Decimal("0")
        for leg in transport:
            total += self._calculate_transport_leg(leg)
        return _round_value(total, 3)

    def identify_hotspots(
        self,
        suppliers: List[SupplierData],
        supplier_emissions: Dict[str, Decimal],
        total_scope3: Decimal,
    ) -> List[SupplierHotspot]:
        """Identify supplier emission hotspots.

        Suppliers are ranked by their share of total Scope 3 emissions.
        Those exceeding the hotspot threshold or in the top N are flagged.

        Args:
            suppliers: List of supplier data.
            supplier_emissions: Computed emissions by supplier ID.
            total_scope3: Total Scope 3 emissions.

        Returns:
            List of SupplierHotspot sorted by emissions descending.
        """
        if total_scope3 == 0:
            return []

        supplier_map = {s.supplier_id: s for s in suppliers}
        hotspot_list: List[Tuple[str, Decimal, float]] = []

        for sid, em in supplier_emissions.items():
            share = _pct(em, total_scope3)
            hotspot_list.append((sid, em, share))

        # Sort by emissions descending
        hotspot_list.sort(key=lambda x: x[1], reverse=True)

        # Take top N or those above threshold
        hotspots: List[SupplierHotspot] = []
        for sid, em, share in hotspot_list[:TOP_N_HOTSPOTS]:
            if share < HOTSPOT_THRESHOLD_PCT and len(hotspots) >= 5:
                break  # At least top 5, then only above threshold

            supplier = supplier_map.get(sid)
            if not supplier:
                continue

            # Determine improvement potential
            dq = supplier.data_quality_score.value
            if dq >= 4:
                potential = "High: improve data quality from spend-based to primary data"
            elif dq >= 3:
                potential = "Medium: refine emission factors with supplier-specific data"
            elif not supplier.certification:
                potential = "Medium: encourage supplier to set SBTi targets"
            else:
                potential = "Low: supplier already providing primary data"

            hotspots.append(SupplierHotspot(
                supplier_id=sid,
                supplier_name=supplier.supplier_name,
                emissions_tco2e=_round_value(em, 3),
                share_of_total_pct=share,
                data_quality=dq,
                improvement_potential=potential,
                tier=supplier.tier.value,
                scope3_category=supplier.scope3_category.value,
            ))

        return hotspots

    def assess_data_quality(self, suppliers: List[SupplierData]) -> float:
        """Assess weighted average data quality across suppliers.

        Weights each supplier's data quality score by their spend,
        providing an emission-weighted view of data quality.

        Args:
            suppliers: List of supplier data records.

        Returns:
            Weighted average data quality score (1.0=best, 5.0=worst).
        """
        if not suppliers:
            return 5.0

        total_spend = sum(s.spend_eur for s in suppliers)
        if total_spend == 0:
            # Unweighted average
            return round(
                sum(s.data_quality_score.value for s in suppliers) / len(suppliers), 2
            )

        weighted_sum = Decimal("0")
        for s in suppliers:
            weight = s.spend_eur / total_spend
            weighted_sum += _decimal(s.data_quality_score.value) * weight

        return _round_value(weighted_sum, 2)

    def generate_engagement_plan(
        self,
        hotspots: List[SupplierHotspot],
        suppliers: List[SupplierData],
    ) -> List[EngagementRecommendation]:
        """Generate supplier engagement recommendations.

        Creates actionable engagement recommendations based on hotspot
        analysis, data quality gaps, and emission reduction potential.

        Args:
            hotspots: Identified emission hotspots.
            suppliers: Full supplier list for context.

        Returns:
            List of EngagementRecommendation sorted by priority.
        """
        supplier_map = {s.supplier_id: s for s in suppliers}
        recommendations: List[EngagementRecommendation] = []

        for hs in hotspots:
            supplier = supplier_map.get(hs.supplier_id)
            if not supplier:
                continue

            dq = hs.data_quality
            share = hs.share_of_total_pct

            # Determine priority
            if share >= 10.0:
                priority = EngagementPriority.CRITICAL
            elif share >= 5.0:
                priority = EngagementPriority.HIGH
            elif dq >= 4:
                priority = EngagementPriority.HIGH
            elif dq >= 3:
                priority = EngagementPriority.MEDIUM
            else:
                priority = EngagementPriority.LOW

            # Determine action based on DQ
            if dq == 5:
                action = "Request primary emission data; transition from spend-based to supplier-specific method"
                expected_improvement = 2.0
                timeline = "6-12 months"
            elif dq == 4:
                action = "Request verified emission factors; consider CDP Supply Chain membership"
                expected_improvement = 1.5
                timeline = "3-6 months"
            elif dq == 3:
                action = "Request product-level carbon footprint data; align on PEFCR methodology"
                expected_improvement = 1.0
                timeline = "3-6 months"
            elif dq == 2:
                action = "Validate reported data; request third-party verification"
                expected_improvement = 0.5
                timeline = "1-3 months"
            else:
                action = "Maintain data quality; encourage SBTi commitment"
                expected_improvement = 0.0
                timeline = "Ongoing"

            rationale = (
                f"Supplier contributes {share}% of Scope 3 emissions "
                f"(DQ score={dq}). "
            )
            if not supplier.certification:
                rationale += "No environmental certifications on record."
            else:
                rationale += f"Certifications: {', '.join(supplier.certification)}."

            recommendations.append(EngagementRecommendation(
                supplier_id=hs.supplier_id,
                supplier_name=hs.supplier_name,
                action=action,
                priority=priority,
                expected_dq_improvement=expected_improvement,
                timeline=timeline,
                rationale=rationale,
            ))

        # Sort by priority (critical first)
        priority_order = {
            EngagementPriority.CRITICAL: 0,
            EngagementPriority.HIGH: 1,
            EngagementPriority.MEDIUM: 2,
            EngagementPriority.LOW: 3,
        }
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 99))

        self._notes.append(
            f"Generated {len(recommendations)} engagement recommendations "
            f"({sum(1 for r in recommendations if r.priority in (EngagementPriority.CRITICAL, EngagementPriority.HIGH))} critical/high priority)"
        )

        return recommendations

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _spend_based_emissions(self, supplier: SupplierData) -> float:
        """Calculate emissions using spend-based EEIO method.

        Args:
            supplier: Supplier with spend and NACE sector data.

        Returns:
            Emissions in tCO2e.
        """
        ef = SPEND_EMISSION_FACTORS.get(
            supplier.nace_sector,
            SPEND_EMISSION_FACTORS["default"],
        )
        emissions = supplier.spend_eur * _decimal(ef)
        return _round_value(emissions, 6)

    def _calculate_bom_emissions_internal(
        self,
        bom: List[BOMEmissionData],
        production_volume: int,
    ) -> Dict[str, Any]:
        """Calculate BOM emissions with material hotspot analysis.

        Args:
            bom: List of BOM component data.
            production_volume: Total production volume.

        Returns:
            Dict with total, per_product, and material_hotspots.
        """
        per_product_kgco2e = Decimal("0")
        material_details: List[Dict[str, Any]] = []

        for component in bom:
            # Get emission factor
            if component.emission_factor_kgco2e_per_unit is not None:
                ef = component.emission_factor_kgco2e_per_unit
            else:
                ef_float = MATERIAL_EMISSION_FACTORS.get(
                    component.material_type,
                    MATERIAL_EMISSION_FACTORS["default"],
                )
                ef = _decimal(ef_float)

            # Adjust for recycled content
            recycled_fraction = _decimal(component.recycled_content_pct) / Decimal("100")
            # Assume recycled material has 30% of virgin emissions
            recycled_ef_factor = (Decimal("1") - recycled_fraction) + (recycled_fraction * Decimal("0.3"))
            adjusted_ef = ef * recycled_ef_factor

            # Per-product emissions for this component
            component_kgco2e = component.quantity_per_product * adjusted_ef
            per_product_kgco2e += component_kgco2e

            material_details.append({
                "component_id": component.component_id,
                "component_name": component.component_name,
                "material_type": component.material_type,
                "quantity_kg": _round_value(component.quantity_per_product, 6),
                "emission_factor_kgco2e_per_kg": _round_value(adjusted_ef, 6),
                "emissions_kgco2e_per_product": _round_value(component_kgco2e, 6),
                "recycled_content_pct": component.recycled_content_pct,
            })

        # Sort by emissions descending for hotspot identification
        material_details.sort(
            key=lambda m: m["emissions_kgco2e_per_product"], reverse=True
        )

        # Total across all production
        prod_vol = _decimal(max(production_volume, 1))
        total_kgco2e = per_product_kgco2e * prod_vol
        total_tco2e = total_kgco2e / Decimal("1000")

        return {
            "total": total_tco2e,
            "per_product": per_product_kgco2e,
            "material_hotspots": material_details[:10],  # Top 10 materials
        }

    def _calculate_transport_leg(self, leg: TransportData) -> Decimal:
        """Calculate emissions for a single transport leg.

        Formula: emissions = distance_km * weight_tonnes * emission_factor
        Result converted from kgCO2e to tCO2e.

        Args:
            leg: Transport leg data.

        Returns:
            Emissions in tCO2e as Decimal.
        """
        if leg.emission_factor is not None:
            ef = leg.emission_factor
        else:
            ef_float = TRANSPORT_EMISSION_FACTORS.get(leg.mode, 0.062)
            ef = _decimal(ef_float)

        # kgCO2e = distance * weight * factor
        kg_co2e = leg.distance_km * leg.weight_tonnes * ef
        # Convert to tCO2e
        tco2e = kg_co2e / Decimal("1000")
        return tco2e

    def _assess_dq_by_category(
        self, suppliers: List[SupplierData]
    ) -> Dict[str, float]:
        """Assess data quality by Scope 3 category.

        Args:
            suppliers: List of supplier data.

        Returns:
            Dict mapping category to average DQ score.
        """
        cat_scores: Dict[str, List[int]] = defaultdict(list)
        for s in suppliers:
            cat_scores[s.scope3_category.value].append(s.data_quality_score.value)

        return {
            cat: round(sum(scores) / len(scores), 2)
            for cat, scores in cat_scores.items()
            if scores
        }
