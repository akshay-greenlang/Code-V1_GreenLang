# -*- coding: utf-8 -*-
"""
Scope3ActivityEngine - PACK-022 Net Zero Acceleration Engine 4
================================================================

Activity-based Scope 3 emission calculations for 9 top categories
with hybrid methodology (activity-based primary, spend-based fallback),
data quality scoring, and activity-vs-spend comparison.

This engine upgrades from the spend-based approach in PACK-021 to
an activity-based methodology for the most material Scope 3 categories.
Activity-based uses physical units (kg, tkm, pkm, kWh) with process-
level emission factors, yielding higher-quality estimates (GHG Protocol
data quality score 2-3 vs 4-5 for spend-based).

Supported Categories (Activity-Based):
    Cat 1:  Purchased goods & services (product-level cradle-to-gate LCA)
    Cat 3:  Fuel & energy related activities (WTT + T&D losses)
    Cat 4:  Upstream transportation (tkm by mode)
    Cat 5:  Waste generated in operations (by waste type & treatment)
    Cat 6:  Business travel (distance by mode)
    Cat 7:  Employee commuting (mode-split + remote work)
    Cat 9:  Downstream transportation (tkm by mode for sold products)
    Cat 11: Use of sold products (energy consumption in lifetime)
    Cat 12: End-of-life treatment (waste treatment of sold products)

Remaining categories (2, 8, 10, 13, 14, 15) use spend-based fallback.

Calculation Methodology:
    Activity-based:
        tCO2e = activity_quantity * activity_emission_factor

    Transport (Cat 4, 9):
        tCO2e = tonnes * km * mode_factor_per_tkm / 1_000_000

    Business travel (Cat 6):
        tCO2e = distance_km * mode_factor_per_pkm / 1_000_000

    Commuting (Cat 7):
        tCO2e = employees * avg_distance * working_days * mode_factor
                * (1 - remote_work_pct)

    Use of sold products (Cat 11):
        tCO2e = units_sold * energy_per_use_kwh * uses_per_lifetime
                * grid_factor / 1000

    Data quality score:
        activity_based = 2-3, spend_based = 4-5
        overall = weighted average by emissions

Emission Factor Sources:
    - DEFRA 2024 UK Government GHG Conversion Factors
    - US EPA Emission Factor Hub (2024)
    - Ecoinvent v3.10 summary factors
    - IEA Emission Factors (2024)

Regulatory References:
    - GHG Protocol Scope 3 Standard (2011) - Chapters 1-15
    - GHG Protocol Scope 3 Technical Guidance (2013)
    - GHG Protocol Scope 3 Evaluator Tool
    - SBTi Corporate Net-Zero Standard v1.2 (2023) - Scope 3 requirements
    - ISO 14064-1:2018 - Category-level quantification

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Emission factors are hard-coded constants from authoritative sources
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


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""
    CAT_01 = "cat_01_purchased_goods"
    CAT_02 = "cat_02_capital_goods"
    CAT_03 = "cat_03_fuel_energy"
    CAT_04 = "cat_04_upstream_transport"
    CAT_05 = "cat_05_waste"
    CAT_06 = "cat_06_business_travel"
    CAT_07 = "cat_07_employee_commuting"
    CAT_08 = "cat_08_upstream_leased"
    CAT_09 = "cat_09_downstream_transport"
    CAT_10 = "cat_10_processing_sold"
    CAT_11 = "cat_11_use_of_sold"
    CAT_12 = "cat_12_end_of_life"
    CAT_13 = "cat_13_downstream_leased"
    CAT_14 = "cat_14_franchises"
    CAT_15 = "cat_15_investments"


class CalculationMethod(str, Enum):
    """Calculation methodology used for a category."""
    ACTIVITY_BASED = "activity_based"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"
    SUPPLIER_SPECIFIC = "supplier_specific"


class TransportMode(str, Enum):
    """Transport mode for categories 4 and 9."""
    ROAD_TRUCK = "road_truck"
    ROAD_VAN = "road_van"
    RAIL_FREIGHT = "rail_freight"
    SEA_CONTAINER = "sea_container"
    SEA_BULK = "sea_bulk"
    AIR_FREIGHT = "air_freight"
    INLAND_WATER = "inland_water"


class WasteType(str, Enum):
    """Waste type for category 5."""
    GENERAL_LANDFILL = "general_landfill"
    ORGANIC_LANDFILL = "organic_landfill"
    INCINERATION = "incineration"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    WASTEWATER = "wastewater"
    HAZARDOUS = "hazardous"


class TravelMode(str, Enum):
    """Business travel mode for category 6."""
    AIR_SHORT_HAUL = "air_short_haul"
    AIR_MEDIUM_HAUL = "air_medium_haul"
    AIR_LONG_HAUL = "air_long_haul"
    RAIL = "rail"
    CAR_RENTAL = "car_rental"
    TAXI = "taxi"
    HOTEL_NIGHT = "hotel_night"


class CommuteMode(str, Enum):
    """Commuting mode for category 7."""
    CAR_PETROL = "car_petrol"
    CAR_DIESEL = "car_diesel"
    CAR_ELECTRIC = "car_electric"
    CAR_HYBRID = "car_hybrid"
    BUS = "bus"
    RAIL_COMMUTER = "rail_commuter"
    METRO = "metro"
    BICYCLE = "bicycle"
    WALK = "walk"
    MOTORCYCLE = "motorcycle"


# ---------------------------------------------------------------------------
# Constants -- Emission Factors
# ---------------------------------------------------------------------------

# Transport emission factors (gCO2e per tonne-km).
# Source: DEFRA 2024, GLEC Framework v3.0.
TRANSPORT_FACTORS: Dict[str, Decimal] = {
    TransportMode.ROAD_TRUCK: Decimal("62.0"),
    TransportMode.ROAD_VAN: Decimal("240.0"),
    TransportMode.RAIL_FREIGHT: Decimal("6.1"),
    TransportMode.SEA_CONTAINER: Decimal("7.9"),
    TransportMode.SEA_BULK: Decimal("3.1"),
    TransportMode.AIR_FREIGHT: Decimal("602.0"),
    TransportMode.INLAND_WATER: Decimal("32.0"),
}

# Business travel emission factors (kgCO2e per passenger-km or per event).
# Source: DEFRA 2024.
TRAVEL_FACTORS: Dict[str, Dict[str, Decimal]] = {
    TravelMode.AIR_SHORT_HAUL: {
        "factor": Decimal("0.15845"),
        "unit": "kgCO2e_per_pkm",
    },
    TravelMode.AIR_MEDIUM_HAUL: {
        "factor": Decimal("0.09740"),
        "unit": "kgCO2e_per_pkm",
    },
    TravelMode.AIR_LONG_HAUL: {
        "factor": Decimal("0.11030"),
        "unit": "kgCO2e_per_pkm",
    },
    TravelMode.RAIL: {
        "factor": Decimal("0.03549"),
        "unit": "kgCO2e_per_pkm",
    },
    TravelMode.CAR_RENTAL: {
        "factor": Decimal("0.17140"),
        "unit": "kgCO2e_per_km",
    },
    TravelMode.TAXI: {
        "factor": Decimal("0.20880"),
        "unit": "kgCO2e_per_km",
    },
    TravelMode.HOTEL_NIGHT: {
        "factor": Decimal("20.6"),
        "unit": "kgCO2e_per_night",
    },
}

# Commuting emission factors (kgCO2e per passenger-km).
# Source: DEFRA 2024.
COMMUTE_FACTORS: Dict[str, Decimal] = {
    CommuteMode.CAR_PETROL: Decimal("0.17140"),
    CommuteMode.CAR_DIESEL: Decimal("0.16844"),
    CommuteMode.CAR_ELECTRIC: Decimal("0.04610"),
    CommuteMode.CAR_HYBRID: Decimal("0.11580"),
    CommuteMode.BUS: Decimal("0.10312"),
    CommuteMode.RAIL_COMMUTER: Decimal("0.03549"),
    CommuteMode.METRO: Decimal("0.02781"),
    CommuteMode.BICYCLE: Decimal("0"),
    CommuteMode.WALK: Decimal("0"),
    CommuteMode.MOTORCYCLE: Decimal("0.11337"),
}

# Waste emission factors (kgCO2e per tonne of waste).
# Source: DEFRA 2024, EPA WARM model.
WASTE_FACTORS: Dict[str, Decimal] = {
    WasteType.GENERAL_LANDFILL: Decimal("586.0"),
    WasteType.ORGANIC_LANDFILL: Decimal("1170.0"),
    WasteType.INCINERATION: Decimal("21.3"),
    WasteType.RECYCLING: Decimal("21.3"),
    WasteType.COMPOSTING: Decimal("10.2"),
    WasteType.WASTEWATER: Decimal("230.0"),
    WasteType.HAZARDOUS: Decimal("850.0"),
}

# WTT (well-to-tank) + T&D loss factors for Cat 3.
# Source: DEFRA 2024.
WTT_FACTORS: Dict[str, Decimal] = {
    "electricity_kwh": Decimal("0.01951"),     # kgCO2e per kWh WTT
    "natural_gas_kwh": Decimal("0.02392"),     # kgCO2e per kWh WTT
    "diesel_litre": Decimal("0.60868"),        # kgCO2e per litre WTT
    "gasoline_litre": Decimal("0.55780"),      # kgCO2e per litre WTT
    "td_loss_factor": Decimal("0.01895"),      # kgCO2e per kWh T&D
}

# Cradle-to-gate emission factors for purchased goods (kgCO2e per kg).
# Source: Ecoinvent v3.10 summary, DEFRA 2024.
PRODUCT_FACTORS: Dict[str, Decimal] = {
    "steel": Decimal("1.89"),
    "aluminium": Decimal("8.60"),
    "cement": Decimal("0.91"),
    "plastics_general": Decimal("3.31"),
    "paper_cardboard": Decimal("1.31"),
    "glass": Decimal("0.86"),
    "textiles_cotton": Decimal("5.43"),
    "textiles_polyester": Decimal("6.40"),
    "electronics": Decimal("12.5"),
    "food_meat": Decimal("25.0"),
    "food_dairy": Decimal("3.20"),
    "food_cereals": Decimal("0.51"),
    "food_vegetables": Decimal("0.37"),
    "chemicals": Decimal("2.85"),
    "wood_timber": Decimal("0.31"),
    "concrete": Decimal("0.14"),
    "copper": Decimal("3.81"),
    "rubber": Decimal("3.18"),
}

# Spend-based fallback factors (tCO2e per $1000 USD).
SPEND_FACTORS: Dict[str, Decimal] = {
    Scope3Category.CAT_02: Decimal("0.350"),
    Scope3Category.CAT_08: Decimal("0.240"),
    Scope3Category.CAT_10: Decimal("0.390"),
    Scope3Category.CAT_13: Decimal("0.220"),
    Scope3Category.CAT_14: Decimal("0.270"),
    Scope3Category.CAT_15: Decimal("0.150"),
}

# Data quality scores by method.
DQ_SCORES: Dict[str, int] = {
    CalculationMethod.SUPPLIER_SPECIFIC: 1,
    CalculationMethod.ACTIVITY_BASED: 2,
    CalculationMethod.HYBRID: 3,
    CalculationMethod.SPEND_BASED: 4,
}

# Category display names.
CATEGORY_NAMES: Dict[str, str] = {
    "cat_01_purchased_goods": "Cat 1: Purchased Goods & Services",
    "cat_02_capital_goods": "Cat 2: Capital Goods",
    "cat_03_fuel_energy": "Cat 3: Fuel- & Energy-Related Activities",
    "cat_04_upstream_transport": "Cat 4: Upstream Transportation",
    "cat_05_waste": "Cat 5: Waste Generated in Operations",
    "cat_06_business_travel": "Cat 6: Business Travel",
    "cat_07_employee_commuting": "Cat 7: Employee Commuting",
    "cat_08_upstream_leased": "Cat 8: Upstream Leased Assets",
    "cat_09_downstream_transport": "Cat 9: Downstream Transportation",
    "cat_10_processing_sold": "Cat 10: Processing of Sold Products",
    "cat_11_use_of_sold": "Cat 11: Use of Sold Products",
    "cat_12_end_of_life": "Cat 12: End-of-Life Treatment",
    "cat_13_downstream_leased": "Cat 13: Downstream Leased Assets",
    "cat_14_franchises": "Cat 14: Franchises",
    "cat_15_investments": "Cat 15: Investments",
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class PurchasedGoodEntry(BaseModel):
    """Purchased goods activity data for Cat 1."""
    product_type: str = Field(..., description="Product type key")
    quantity_kg: Decimal = Field(..., ge=Decimal("0"))
    custom_factor_kgco2e_per_kg: Optional[Decimal] = Field(None, ge=Decimal("0"))


class FuelEnergyEntry(BaseModel):
    """Fuel/energy-related data for Cat 3."""
    fuel_type: str = Field(..., description="Fuel type key")
    quantity: Decimal = Field(..., ge=Decimal("0"))
    unit: str = Field(default="kwh", description="kwh or litre")


class TransportEntry(BaseModel):
    """Transport activity data for Cat 4 and 9."""
    mode: TransportMode = Field(...)
    tonnes: Decimal = Field(..., ge=Decimal("0"))
    distance_km: Decimal = Field(..., ge=Decimal("0"))


class WasteEntry(BaseModel):
    """Waste data for Cat 5."""
    waste_type: WasteType = Field(...)
    quantity_tonnes: Decimal = Field(..., ge=Decimal("0"))


class TravelEntry(BaseModel):
    """Business travel data for Cat 6."""
    mode: TravelMode = Field(...)
    distance_km: Optional[Decimal] = Field(None, ge=Decimal("0"))
    nights: Optional[int] = Field(None, ge=0)


class CommuteProfile(BaseModel):
    """Employee commuting profile for Cat 7."""
    mode: CommuteMode = Field(...)
    employee_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    avg_distance_km_one_way: Decimal = Field(default=Decimal("15"), ge=Decimal("0"))


class UseOfSoldEntry(BaseModel):
    """Use-of-sold-products data for Cat 11."""
    product_name: str = Field(default="")
    units_sold: Decimal = Field(..., ge=Decimal("0"))
    energy_per_use_kwh: Decimal = Field(..., ge=Decimal("0"))
    uses_per_lifetime: Decimal = Field(..., ge=Decimal("0"))
    grid_region: str = Field(default="GLOBAL_AVG")


class EndOfLifeEntry(BaseModel):
    """End-of-life treatment data for Cat 12."""
    product_name: str = Field(default="")
    units_sold: Decimal = Field(..., ge=Decimal("0"))
    weight_per_unit_kg: Decimal = Field(..., ge=Decimal("0"))
    treatment_type: WasteType = Field(default=WasteType.GENERAL_LANDFILL)


class SpendFallbackEntry(BaseModel):
    """Spend-based fallback for non-activity categories."""
    category: Scope3Category = Field(...)
    spend_usd_thousands: Decimal = Field(..., ge=Decimal("0"))
    custom_factor: Optional[Decimal] = Field(None, ge=Decimal("0"))


class Scope3ActivityInput(BaseModel):
    """Complete input for activity-based Scope 3 calculation.

    Attributes:
        entity_name: Reporting entity name.
        reporting_year: Reporting year.
        purchased_goods: Cat 1 product-level data.
        fuel_energy: Cat 3 WTT/T&D data.
        upstream_transport: Cat 4 transport data.
        waste: Cat 5 waste data.
        business_travel: Cat 6 travel data.
        commuting_profiles: Cat 7 commuting data.
        total_employees: Total employee headcount.
        working_days_per_year: Working days per year.
        remote_work_pct: Percentage of employees working remotely.
        downstream_transport: Cat 9 transport data.
        use_of_sold_products: Cat 11 data.
        end_of_life: Cat 12 data.
        spend_fallbacks: Spend-based data for remaining categories.
        grid_factor_tco2e_per_mwh: Grid factor for Cat 11.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    reporting_year: int = Field(
        ..., ge=2020, le=2100, description="Reporting year"
    )
    purchased_goods: List[PurchasedGoodEntry] = Field(default_factory=list)
    fuel_energy: List[FuelEnergyEntry] = Field(default_factory=list)
    upstream_transport: List[TransportEntry] = Field(default_factory=list)
    waste: List[WasteEntry] = Field(default_factory=list)
    business_travel: List[TravelEntry] = Field(default_factory=list)
    commuting_profiles: List[CommuteProfile] = Field(default_factory=list)
    total_employees: int = Field(default=0, ge=0)
    working_days_per_year: int = Field(default=230, ge=1, le=365)
    remote_work_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    downstream_transport: List[TransportEntry] = Field(default_factory=list)
    use_of_sold_products: List[UseOfSoldEntry] = Field(default_factory=list)
    end_of_life: List[EndOfLifeEntry] = Field(default_factory=list)
    spend_fallbacks: List[SpendFallbackEntry] = Field(default_factory=list)
    grid_factor_tco2e_per_mwh: Decimal = Field(
        default=Decimal("0.436"), ge=Decimal("0"),
        description="Grid factor for use-phase calculations",
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class CategoryResult(BaseModel):
    """Emission result for a single Scope 3 category.

    Attributes:
        category: Category identifier.
        category_name: Display name.
        emissions_tco2e: Calculated emissions.
        method: Calculation method used.
        data_quality_score: GHG Protocol quality score (1-5).
        pct_of_total: Percentage of total Scope 3.
        is_material: Whether category is material (>5% or >1% of total).
        entry_count: Number of data entries for this category.
    """
    category: str = Field(default="")
    category_name: str = Field(default="")
    emissions_tco2e: Decimal = Field(default=Decimal("0"))
    method: str = Field(default=CalculationMethod.SPEND_BASED.value)
    data_quality_score: int = Field(default=5)
    pct_of_total: Decimal = Field(default=Decimal("0"))
    is_material: bool = Field(default=False)
    entry_count: int = Field(default=0)


class MethodComparison(BaseModel):
    """Comparison between activity-based and spend-based results.

    Attributes:
        category: Category identifier.
        activity_based_tco2e: Activity-based result.
        spend_based_tco2e: Spend-based result (if available).
        delta_tco2e: Absolute difference.
        delta_pct: Percentage difference.
        materiality_flag: Whether the difference is material (>25%).
    """
    category: str = Field(default="")
    activity_based_tco2e: Decimal = Field(default=Decimal("0"))
    spend_based_tco2e: Decimal = Field(default=Decimal("0"))
    delta_tco2e: Decimal = Field(default=Decimal("0"))
    delta_pct: Decimal = Field(default=Decimal("0"))
    materiality_flag: bool = Field(default=False)


class Scope3ActivityResult(BaseModel):
    """Complete activity-based Scope 3 result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        reporting_year: Reporting year.
        category_emissions: Per-category results.
        total_scope3_tco2e: Total Scope 3 emissions.
        upstream_tco2e: Total upstream (Cat 1-8).
        downstream_tco2e: Total downstream (Cat 9-15).
        activity_based_pct: Percentage of total from activity-based.
        spend_based_pct: Percentage from spend-based fallback.
        overall_data_quality: Weighted average quality score.
        method_comparisons: Activity vs spend comparisons.
        material_categories: Categories flagged as material.
        categories_count: Number of categories with data.
        recommendations: Improvement recommendations.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=0)
    category_emissions: List[CategoryResult] = Field(default_factory=list)
    total_scope3_tco2e: Decimal = Field(default=Decimal("0"))
    upstream_tco2e: Decimal = Field(default=Decimal("0"))
    downstream_tco2e: Decimal = Field(default=Decimal("0"))
    activity_based_pct: Decimal = Field(default=Decimal("0"))
    spend_based_pct: Decimal = Field(default=Decimal("0"))
    overall_data_quality: Decimal = Field(default=Decimal("0"))
    method_comparisons: List[MethodComparison] = Field(default_factory=list)
    material_categories: List[str] = Field(default_factory=list)
    categories_count: int = Field(default=0)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Scope3ActivityEngine:
    """Activity-based Scope 3 emission calculation engine.

    Calculates emissions for 9 categories using activity data with
    physical emission factors, and falls back to spend-based for
    remaining categories.

    All calculations use Decimal arithmetic.  No LLM in any path.

    Usage::

        engine = Scope3ActivityEngine()
        result = engine.calculate(scope3_input)
        print(f"Total Scope 3: {result.total_scope3_tco2e} tCO2e")
    """

    engine_version: str = _MODULE_VERSION

    UPSTREAM_CATEGORIES = {
        Scope3Category.CAT_01.value, Scope3Category.CAT_02.value,
        Scope3Category.CAT_03.value, Scope3Category.CAT_04.value,
        Scope3Category.CAT_05.value, Scope3Category.CAT_06.value,
        Scope3Category.CAT_07.value, Scope3Category.CAT_08.value,
    }

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: Scope3ActivityInput) -> Scope3ActivityResult:
        """Run complete Scope 3 activity-based calculation.

        Args:
            data: Validated Scope 3 input.

        Returns:
            Scope3ActivityResult with all 15 categories.
        """
        t0 = time.perf_counter()
        logger.info(
            "Scope 3 activity: entity=%s, year=%d",
            data.entity_name, data.reporting_year,
        )

        results: Dict[str, CategoryResult] = {}

        # Cat 1: Purchased Goods
        results[Scope3Category.CAT_01.value] = self._calc_cat01(data)

        # Cat 3: Fuel & Energy
        results[Scope3Category.CAT_03.value] = self._calc_cat03(data)

        # Cat 4: Upstream Transport
        results[Scope3Category.CAT_04.value] = self._calc_cat04(data)

        # Cat 5: Waste
        results[Scope3Category.CAT_05.value] = self._calc_cat05(data)

        # Cat 6: Business Travel
        results[Scope3Category.CAT_06.value] = self._calc_cat06(data)

        # Cat 7: Employee Commuting
        results[Scope3Category.CAT_07.value] = self._calc_cat07(data)

        # Cat 9: Downstream Transport
        results[Scope3Category.CAT_09.value] = self._calc_cat09(data)

        # Cat 11: Use of Sold Products
        results[Scope3Category.CAT_11.value] = self._calc_cat11(data)

        # Cat 12: End-of-Life
        results[Scope3Category.CAT_12.value] = self._calc_cat12(data)

        # Spend-based fallback for remaining categories
        for entry in data.spend_fallbacks:
            cat_val = entry.category.value
            if cat_val not in results or results[cat_val].emissions_tco2e == Decimal("0"):
                results[cat_val] = self._calc_spend_fallback(entry)

        # Ensure all 15 categories present
        for cat in Scope3Category:
            if cat.value not in results:
                results[cat.value] = CategoryResult(
                    category=cat.value,
                    category_name=CATEGORY_NAMES.get(cat.value, cat.value),
                    emissions_tco2e=Decimal("0"),
                    method=CalculationMethod.SPEND_BASED.value,
                    data_quality_score=5,
                )

        # Compute totals
        all_results = list(results.values())
        total = sum(r.emissions_tco2e for r in all_results)

        # Set percentages and materiality
        for r in all_results:
            r.pct_of_total = _round_val(_safe_pct(r.emissions_tco2e, total), 2)
            r.is_material = r.pct_of_total >= Decimal("1")

        upstream = sum(
            r.emissions_tco2e for r in all_results
            if r.category in self.UPSTREAM_CATEGORIES
        )
        downstream = total - upstream

        # Activity-based vs spend-based split
        activity_em = sum(
            r.emissions_tco2e for r in all_results
            if r.method == CalculationMethod.ACTIVITY_BASED.value
        )
        spend_em = sum(
            r.emissions_tco2e for r in all_results
            if r.method == CalculationMethod.SPEND_BASED.value
        )

        # Data quality weighted average
        dq_weighted_sum = Decimal("0")
        dq_weight_total = Decimal("0")
        for r in all_results:
            if r.emissions_tco2e > Decimal("0"):
                dq_weighted_sum += _decimal(r.data_quality_score) * r.emissions_tco2e
                dq_weight_total += r.emissions_tco2e
        overall_dq = _safe_divide(dq_weighted_sum, dq_weight_total)

        # Material categories
        material_cats = [
            r.category for r in all_results if r.is_material
        ]

        # Sort by emissions descending
        all_results.sort(key=lambda r: r.emissions_tco2e, reverse=True)

        # Recommendations
        recommendations = self._generate_recommendations(
            all_results, total, overall_dq
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = Scope3ActivityResult(
            entity_name=data.entity_name,
            reporting_year=data.reporting_year,
            category_emissions=all_results,
            total_scope3_tco2e=_round_val(total),
            upstream_tco2e=_round_val(upstream),
            downstream_tco2e=_round_val(downstream),
            activity_based_pct=_round_val(_safe_pct(activity_em, total), 2),
            spend_based_pct=_round_val(_safe_pct(spend_em, total), 2),
            overall_data_quality=_round_val(overall_dq, 2),
            material_categories=sorted(material_cats),
            categories_count=sum(
                1 for r in all_results if r.emissions_tco2e > Decimal("0")
            ),
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Scope 3 complete: total=%.2f tCO2e, activity=%.1f%%, "
            "quality=%.1f",
            float(total), float(result.activity_based_pct),
            float(overall_dq),
        )
        return result

    # ------------------------------------------------------------------ #
    # Cat 1: Purchased Goods                                              #
    # ------------------------------------------------------------------ #

    def _calc_cat01(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 1 using cradle-to-gate product factors.

        tCO2e = quantity_kg * factor_kgCO2e_per_kg / 1000
        """
        total = Decimal("0")
        for entry in data.purchased_goods:
            factor = entry.custom_factor_kgco2e_per_kg
            if factor is None:
                factor = PRODUCT_FACTORS.get(
                    entry.product_type, Decimal("1.0")
                )
            total += entry.quantity_kg * factor / Decimal("1000")

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.purchased_goods else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_01.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_01.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.purchased_goods),
        )

    # ------------------------------------------------------------------ #
    # Cat 3: Fuel & Energy                                                #
    # ------------------------------------------------------------------ #

    def _calc_cat03(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 3 using WTT + T&D loss factors.

        tCO2e = quantity * wtt_factor / 1000
        """
        total = Decimal("0")
        for entry in data.fuel_energy:
            factor_key = f"{entry.fuel_type}_{entry.unit}"
            factor = WTT_FACTORS.get(factor_key, Decimal("0.02"))
            total += entry.quantity * factor / Decimal("1000")
            # Add T&D losses for electricity
            if "electricity" in entry.fuel_type:
                total += (
                    entry.quantity
                    * WTT_FACTORS["td_loss_factor"]
                    / Decimal("1000")
                )

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.fuel_energy else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_03.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_03.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.fuel_energy),
        )

    # ------------------------------------------------------------------ #
    # Cat 4: Upstream Transport                                           #
    # ------------------------------------------------------------------ #

    def _calc_cat04(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 4 using tonne-km and mode-specific factors.

        tCO2e = tonnes * km * factor_gCO2e_per_tkm / 1_000_000
        """
        total = Decimal("0")
        for entry in data.upstream_transport:
            factor = TRANSPORT_FACTORS.get(entry.mode, Decimal("62.0"))
            total += (
                entry.tonnes * entry.distance_km * factor
                / Decimal("1000000")
            )

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.upstream_transport else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_04.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_04.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.upstream_transport),
        )

    # ------------------------------------------------------------------ #
    # Cat 5: Waste                                                        #
    # ------------------------------------------------------------------ #

    def _calc_cat05(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 5 using waste type and treatment factors.

        tCO2e = tonnes * factor_kgCO2e_per_tonne / 1000
        """
        total = Decimal("0")
        for entry in data.waste:
            factor = WASTE_FACTORS.get(entry.waste_type, Decimal("586.0"))
            total += entry.quantity_tonnes * factor / Decimal("1000")

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.waste else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_05.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_05.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.waste),
        )

    # ------------------------------------------------------------------ #
    # Cat 6: Business Travel                                              #
    # ------------------------------------------------------------------ #

    def _calc_cat06(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 6 using distance and mode factors.

        Air/Rail/Car: tCO2e = distance_km * factor / 1000
        Hotel: tCO2e = nights * factor / 1000
        """
        total = Decimal("0")
        for entry in data.business_travel:
            factor_data = TRAVEL_FACTORS.get(entry.mode)
            if factor_data is None:
                continue
            factor = factor_data["factor"]
            if entry.mode == TravelMode.HOTEL_NIGHT:
                nights = _decimal(entry.nights or 0)
                total += nights * factor / Decimal("1000")
            else:
                distance = entry.distance_km or Decimal("0")
                total += distance * factor / Decimal("1000")

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.business_travel else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_06.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_06.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.business_travel),
        )

    # ------------------------------------------------------------------ #
    # Cat 7: Employee Commuting                                           #
    # ------------------------------------------------------------------ #

    def _calc_cat07(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 7 using mode-split with remote work adjustment.

        tCO2e = employees * pct * avg_dist * 2 * working_days * factor
                * (1 - remote_pct) / 1000
        """
        total = Decimal("0")
        employees = _decimal(data.total_employees)
        working_days = _decimal(data.working_days_per_year)
        remote_adj = Decimal("1") - data.remote_work_pct / Decimal("100")

        for profile in data.commuting_profiles:
            factor = COMMUTE_FACTORS.get(profile.mode, Decimal("0.10"))
            emp_count = employees * profile.employee_pct / Decimal("100")
            round_trip = profile.avg_distance_km_one_way * Decimal("2")

            annual_em = (
                emp_count * round_trip * working_days * factor
                * remote_adj / Decimal("1000")
            )
            total += annual_em

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.commuting_profiles else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_07.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_07.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.commuting_profiles),
        )

    # ------------------------------------------------------------------ #
    # Cat 9: Downstream Transport                                         #
    # ------------------------------------------------------------------ #

    def _calc_cat09(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 9 using same methodology as Cat 4."""
        total = Decimal("0")
        for entry in data.downstream_transport:
            factor = TRANSPORT_FACTORS.get(entry.mode, Decimal("62.0"))
            total += (
                entry.tonnes * entry.distance_km * factor
                / Decimal("1000000")
            )

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.downstream_transport else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_09.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_09.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.downstream_transport),
        )

    # ------------------------------------------------------------------ #
    # Cat 11: Use of Sold Products                                        #
    # ------------------------------------------------------------------ #

    def _calc_cat11(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 11 using energy consumption during product life.

        tCO2e = units * energy_per_use * uses * grid_factor / 1000
        """
        total = Decimal("0")
        grid_factor = data.grid_factor_tco2e_per_mwh

        for entry in data.use_of_sold_products:
            total_kwh = (
                entry.units_sold * entry.energy_per_use_kwh
                * entry.uses_per_lifetime
            )
            # kWh to MWh
            total_mwh = total_kwh / Decimal("1000")
            total += total_mwh * grid_factor

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.use_of_sold_products else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_11.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_11.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.use_of_sold_products),
        )

    # ------------------------------------------------------------------ #
    # Cat 12: End-of-Life                                                 #
    # ------------------------------------------------------------------ #

    def _calc_cat12(self, data: Scope3ActivityInput) -> CategoryResult:
        """Calculate Cat 12 using waste treatment of sold products.

        tCO2e = units * weight_kg * waste_factor / 1_000_000
        """
        total = Decimal("0")
        for entry in data.end_of_life:
            factor = WASTE_FACTORS.get(
                entry.treatment_type, Decimal("586.0")
            )
            total_kg = entry.units_sold * entry.weight_per_unit_kg
            total += total_kg * factor / Decimal("1000000")

        method = (
            CalculationMethod.ACTIVITY_BASED.value
            if data.end_of_life else CalculationMethod.SPEND_BASED.value
        )

        return CategoryResult(
            category=Scope3Category.CAT_12.value,
            category_name=CATEGORY_NAMES[Scope3Category.CAT_12.value],
            emissions_tco2e=_round_val(total),
            method=method,
            data_quality_score=DQ_SCORES.get(method, 4),
            entry_count=len(data.end_of_life),
        )

    # ------------------------------------------------------------------ #
    # Spend-Based Fallback                                                #
    # ------------------------------------------------------------------ #

    def _calc_spend_fallback(
        self, entry: SpendFallbackEntry
    ) -> CategoryResult:
        """Calculate emissions using spend-based methodology.

        tCO2e = spend_thousands * factor
        """
        factor = entry.custom_factor
        if factor is None:
            factor = SPEND_FACTORS.get(entry.category, Decimal("0.300"))
        total = entry.spend_usd_thousands * factor

        return CategoryResult(
            category=entry.category.value,
            category_name=CATEGORY_NAMES.get(
                entry.category.value, entry.category.value
            ),
            emissions_tco2e=_round_val(total),
            method=CalculationMethod.SPEND_BASED.value,
            data_quality_score=DQ_SCORES[CalculationMethod.SPEND_BASED],
            entry_count=1,
        )

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        results: List[CategoryResult],
        total: Decimal,
        overall_dq: Decimal,
    ) -> List[str]:
        """Generate data improvement recommendations.

        Args:
            results: All category results.
            total: Total Scope 3 emissions.
            overall_dq: Overall data quality score.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # High spend-based reliance
        spend_cats = [
            r for r in results
            if r.method == CalculationMethod.SPEND_BASED.value
            and r.emissions_tco2e > Decimal("0")
        ]
        if spend_cats:
            spend_pct = _safe_pct(
                sum(r.emissions_tco2e for r in spend_cats), total
            )
            if spend_pct > Decimal("50"):
                recs.append(
                    f"{spend_pct}% of Scope 3 relies on spend-based "
                    "estimates. Upgrade to activity-based data for top "
                    "categories to improve accuracy."
                )

        # Top category suggestions
        material = [r for r in results if r.is_material]
        for r in material[:3]:
            if r.method == CalculationMethod.SPEND_BASED.value:
                recs.append(
                    f"{r.category_name} ({r.pct_of_total}% of total) uses "
                    "spend-based methodology. Collect activity data from "
                    "suppliers to upgrade to activity-based."
                )

        # Data quality
        if overall_dq > Decimal("3.5"):
            recs.append(
                f"Overall data quality score is {overall_dq}/5. "
                "Target score <3.0 by transitioning top categories "
                "to activity-based or supplier-specific data."
            )

        # Cat 7 remote work
        cat7 = next(
            (r for r in results if r.category == Scope3Category.CAT_07.value),
            None,
        )
        if cat7 and cat7.entry_count == 0:
            recs.append(
                "Employee commuting (Cat 7) has no data. Conduct a "
                "commuting survey to establish mode-split data."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_emission_factor(
        self, factor_type: str, key: str
    ) -> Optional[str]:
        """Look up an emission factor by type and key.

        Args:
            factor_type: Factor type ('transport', 'travel', 'waste', 'product').
            key: Factor key.

        Returns:
            Factor value as string, or None if not found.
        """
        if factor_type == "transport":
            val = TRANSPORT_FACTORS.get(key)
        elif factor_type == "travel":
            data = TRAVEL_FACTORS.get(key)
            val = data["factor"] if data else None
        elif factor_type == "waste":
            val = WASTE_FACTORS.get(key)
        elif factor_type == "product":
            val = PRODUCT_FACTORS.get(key)
        elif factor_type == "commute":
            val = COMMUTE_FACTORS.get(key)
        else:
            val = None

        return str(val) if val is not None else None

    def get_summary(
        self, result: Scope3ActivityResult
    ) -> Dict[str, Any]:
        """Generate concise Scope 3 summary.

        Args:
            result: Result to summarize.

        Returns:
            Dict with key metrics and provenance hash.
        """
        top3 = sorted(
            result.category_emissions,
            key=lambda r: r.emissions_tco2e,
            reverse=True,
        )[:3]

        summary: Dict[str, Any] = {
            "entity_name": result.entity_name,
            "reporting_year": result.reporting_year,
            "total_scope3_tco2e": str(result.total_scope3_tco2e),
            "upstream_tco2e": str(result.upstream_tco2e),
            "downstream_tco2e": str(result.downstream_tco2e),
            "activity_based_pct": str(result.activity_based_pct),
            "data_quality": str(result.overall_data_quality),
            "top_3_categories": [
                {
                    "name": c.category_name,
                    "tco2e": str(c.emissions_tco2e),
                    "pct": str(c.pct_of_total),
                }
                for c in top3
            ],
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary
