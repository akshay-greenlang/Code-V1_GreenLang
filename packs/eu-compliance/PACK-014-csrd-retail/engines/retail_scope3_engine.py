# -*- coding: utf-8 -*-
"""
RetailScope3Engine - PACK-014 CSRD Retail Engine 2
====================================================

All 15 Scope 3 categories with retail-specific prioritization and
calculation methods. Supports spend-based, average-data, hybrid, and
supplier-specific approaches with data-quality scoring and hotspot analysis.

Scope 3 Categories (GHG Protocol Corporate Value Chain Standard):
    - Cat 1:  Purchased Goods & Services (CRITICAL for retail)
    - Cat 2:  Capital Goods
    - Cat 3:  Fuel- and Energy-Related Activities
    - Cat 4:  Upstream Transportation & Distribution (CRITICAL for retail)
    - Cat 5:  Waste Generated in Operations
    - Cat 6:  Business Travel
    - Cat 7:  Employee Commuting
    - Cat 8:  Upstream Leased Assets
    - Cat 9:  Downstream Transportation & Distribution (HIGH for e-commerce)
    - Cat 10: Processing of Sold Products
    - Cat 11: Use of Sold Products (HIGH for electronics retailers)
    - Cat 12: End-of-Life Treatment of Sold Products
    - Cat 13: Downstream Leased Assets
    - Cat 14: Franchises
    - Cat 15: Investments

Calculation Methods:
    - SUPPLIER_SPECIFIC: Verified supplier-level data (Score 1-2)
    - HYBRID: Product-level averages + supplier adjustments (Score 3)
    - AVERAGE_DATA: Sector/product-level emission factors (Score 3-4)
    - SPEND_BASED: Financial EEIO emission factors (Score 4-5)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Emission factors from DEFRA, ecoinvent, EEIO databases (hard-coded)
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-014 CSRD Retail & Consumer Goods
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

engine_version: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    # Exclude volatile fields to guarantee bit-perfect reproducibility
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
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
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories."""
    CAT_1 = "cat_1_purchased_goods_services"
    CAT_2 = "cat_2_capital_goods"
    CAT_3 = "cat_3_fuel_energy_activities"
    CAT_4 = "cat_4_upstream_transport"
    CAT_5 = "cat_5_waste_operations"
    CAT_6 = "cat_6_business_travel"
    CAT_7 = "cat_7_employee_commuting"
    CAT_8 = "cat_8_upstream_leased"
    CAT_9 = "cat_9_downstream_transport"
    CAT_10 = "cat_10_processing_sold"
    CAT_11 = "cat_11_use_of_sold"
    CAT_12 = "cat_12_end_of_life"
    CAT_13 = "cat_13_downstream_leased"
    CAT_14 = "cat_14_franchises"
    CAT_15 = "cat_15_investments"

class CalculationMethod(str, Enum):
    """Emission calculation methodology."""
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"

class DataQualityLevel(str, Enum):
    """Data quality scoring per GHG Protocol guidance."""
    SCORE_1 = "score_1"  # Supplier-specific, verified
    SCORE_2 = "score_2"  # Supplier-specific, unverified
    SCORE_3 = "score_3"  # Product-level average
    SCORE_4 = "score_4"  # Sector average
    SCORE_5 = "score_5"  # Spend-based proxy

class ProductCategory(str, Enum):
    """Retail product category classification."""
    FOOD_FRESH = "food_fresh"
    FOOD_PACKAGED = "food_packaged"
    BEVERAGES = "beverages"
    APPAREL = "apparel"
    ELECTRONICS = "electronics"
    FURNITURE = "furniture"
    COSMETICS = "cosmetics"
    HOUSEHOLD = "household"
    DIY = "diy"
    STATIONERY = "stationery"
    TOYS = "toys"
    SPORTS = "sports"
    AUTOMOTIVE_PARTS = "automotive_parts"
    PET_PRODUCTS = "pet_products"
    OTHER = "other"

class TransportMode(str, Enum):
    """Freight transport mode."""
    ROAD = "road"
    RAIL = "rail"
    SEA = "sea"
    AIR = "air"
    INLAND_WATERWAY = "inland_waterway"
    LAST_MILE_VAN = "last_mile_van"
    LAST_MILE_CARGO_BIKE = "last_mile_cargo_bike"

class WasteDisposalMethod(str, Enum):
    """Waste disposal method."""
    LANDFILL = "landfill"
    INCINERATION = "incineration"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"

class TravelMode(str, Enum):
    """Business travel mode."""
    AIR_SHORT = "air_short_haul"
    AIR_MEDIUM = "air_medium_haul"
    AIR_LONG = "air_long_haul"
    RAIL = "rail"
    CAR = "car"
    HOTEL = "hotel"

class CommuteMode(str, Enum):
    """Employee commuting mode."""
    CAR_PETROL = "car_petrol"
    CAR_DIESEL = "car_diesel"
    CAR_HYBRID = "car_hybrid"
    CAR_ELECTRIC = "car_electric"
    BUS = "bus"
    RAIL = "rail"
    BICYCLE = "bicycle"
    WALKING = "walking"
    REMOTE = "remote"

# ---------------------------------------------------------------------------
# Constants -- Emission Factors
# ---------------------------------------------------------------------------

# Spend-based emission factors (tCO2e per EUR million)
# Source: Exiobase / EEIO databases, EU adjusted 2024
SPEND_EMISSION_FACTORS: Dict[str, float] = {
    ProductCategory.FOOD_FRESH: 1850.0,
    ProductCategory.FOOD_PACKAGED: 1420.0,
    ProductCategory.BEVERAGES: 950.0,
    ProductCategory.APPAREL: 820.0,
    ProductCategory.ELECTRONICS: 680.0,
    ProductCategory.FURNITURE: 560.0,
    ProductCategory.COSMETICS: 490.0,
    ProductCategory.HOUSEHOLD: 620.0,
    ProductCategory.DIY: 710.0,
    ProductCategory.STATIONERY: 380.0,
    ProductCategory.TOYS: 520.0,
    ProductCategory.SPORTS: 580.0,
    ProductCategory.AUTOMOTIVE_PARTS: 750.0,
    ProductCategory.PET_PRODUCTS: 640.0,
    ProductCategory.OTHER: 600.0,
    # Additional sub-categories for granularity
    "dairy": 2100.0,
    "meat_beef": 3200.0,
    "meat_poultry": 1800.0,
    "meat_pork": 2100.0,
    "seafood": 1600.0,
    "fruits_vegetables": 850.0,
    "bakery": 720.0,
    "snacks_confectionery": 880.0,
    "frozen_food": 1300.0,
    "canned_goods": 680.0,
    "soft_drinks": 650.0,
    "alcoholic_beverages": 1100.0,
    "fast_fashion": 1050.0,
    "premium_apparel": 620.0,
    "footwear": 780.0,
    "consumer_electronics": 750.0,
    "white_goods": 580.0,
    "lighting": 420.0,
    "cleaning_products": 540.0,
    "personal_care": 460.0,
    "gardening": 480.0,
    "paint_coatings": 820.0,
    "building_materials": 940.0,
    "office_supplies": 350.0,
    "packaging_materials": 1200.0,
    "pharmaceuticals": 310.0,
}

# Product-level emission factors (tCO2e per kg or per unit)
# Source: DEFRA 2024, ecoinvent 3.10, EU PEF
PRODUCT_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    # Food (tCO2e per kg)
    "beef": {"factor": 0.0272, "unit": "kg"},
    "lamb": {"factor": 0.0254, "unit": "kg"},
    "pork": {"factor": 0.0074, "unit": "kg"},
    "poultry": {"factor": 0.0058, "unit": "kg"},
    "fish_farmed": {"factor": 0.0085, "unit": "kg"},
    "fish_wild": {"factor": 0.0046, "unit": "kg"},
    "dairy_milk": {"factor": 0.0032, "unit": "litre"},
    "dairy_cheese": {"factor": 0.0128, "unit": "kg"},
    "eggs": {"factor": 0.0045, "unit": "kg"},
    "rice": {"factor": 0.0040, "unit": "kg"},
    "wheat_flour": {"factor": 0.0012, "unit": "kg"},
    "vegetables": {"factor": 0.0008, "unit": "kg"},
    "fruit_temperate": {"factor": 0.0005, "unit": "kg"},
    "fruit_tropical": {"factor": 0.0012, "unit": "kg"},
    "coffee": {"factor": 0.0168, "unit": "kg"},
    "chocolate": {"factor": 0.0195, "unit": "kg"},
    "sugar": {"factor": 0.0018, "unit": "kg"},
    "olive_oil": {"factor": 0.0062, "unit": "litre"},
    "palm_oil": {"factor": 0.0078, "unit": "litre"},
    # Apparel (tCO2e per unit)
    "cotton_tshirt": {"factor": 0.0070, "unit": "unit"},
    "polyester_garment": {"factor": 0.0058, "unit": "unit"},
    "denim_jeans": {"factor": 0.0335, "unit": "unit"},
    "wool_sweater": {"factor": 0.0185, "unit": "unit"},
    "leather_shoes": {"factor": 0.0140, "unit": "pair"},
    "synthetic_shoes": {"factor": 0.0085, "unit": "pair"},
    # Electronics (tCO2e per unit)
    "smartphone": {"factor": 0.0700, "unit": "unit"},
    "laptop": {"factor": 0.3500, "unit": "unit"},
    "television": {"factor": 0.5500, "unit": "unit"},
    "washing_machine": {"factor": 0.2800, "unit": "unit"},
    "refrigerator": {"factor": 0.3200, "unit": "unit"},
    "led_bulb": {"factor": 0.0025, "unit": "unit"},
}

# Transport emission factors (tCO2e per tonne-km)
# Source: DEFRA 2024, EEA transport statistics
TRANSPORT_EMISSION_FACTORS: Dict[str, float] = {
    TransportMode.ROAD: 0.000062,          # 62 gCO2e/tkm
    TransportMode.RAIL: 0.000022,          # 22 gCO2e/tkm
    TransportMode.SEA: 0.000008,           # 8 gCO2e/tkm (container)
    TransportMode.AIR: 0.000602,           # 602 gCO2e/tkm
    TransportMode.INLAND_WATERWAY: 0.000031,  # 31 gCO2e/tkm
    TransportMode.LAST_MILE_VAN: 0.000248,    # 248 gCO2e/tkm (vkm approx)
    TransportMode.LAST_MILE_CARGO_BIKE: 0.0,  # Zero
}

# Waste emission factors (tCO2e per tonne of waste)
# Source: DEFRA 2024, IPCC waste sector
WASTE_EMISSION_FACTORS: Dict[str, float] = {
    WasteDisposalMethod.LANDFILL: 0.586,
    WasteDisposalMethod.INCINERATION: 0.021,
    WasteDisposalMethod.RECYCLING: 0.021,    # Net after avoided virgin
    WasteDisposalMethod.COMPOSTING: 0.010,
    WasteDisposalMethod.ANAEROBIC_DIGESTION: 0.008,
}

# Business travel emission factors (tCO2e per passenger-km)
# Source: DEFRA 2024
TRAVEL_EMISSION_FACTORS: Dict[str, float] = {
    TravelMode.AIR_SHORT: 0.000156,    # <500 km
    TravelMode.AIR_MEDIUM: 0.000131,   # 500-3700 km
    TravelMode.AIR_LONG: 0.000102,     # >3700 km
    TravelMode.RAIL: 0.000035,
    TravelMode.CAR: 0.000171,
    TravelMode.HOTEL: 0.021,           # Per night
}

# Employee commuting emission factors (tCO2e per passenger-km)
# Source: DEFRA 2024
COMMUTE_EMISSION_FACTORS: Dict[str, float] = {
    CommuteMode.CAR_PETROL: 0.000171,
    CommuteMode.CAR_DIESEL: 0.000168,
    CommuteMode.CAR_HYBRID: 0.000120,
    CommuteMode.CAR_ELECTRIC: 0.000050,  # Scope 2 proxy
    CommuteMode.BUS: 0.000089,
    CommuteMode.RAIL: 0.000035,
    CommuteMode.BICYCLE: 0.0,
    CommuteMode.WALKING: 0.0,
    CommuteMode.REMOTE: 0.000008,  # Home office energy per hour equiv.
}

# Use-phase electricity for product categories (kWh per year typical use)
# Source: EU Energy Label data, ecoinvent
USE_PHASE_ELECTRICITY: Dict[str, Dict[str, float]] = {
    "smartphone": {"kwh_per_year": 4.0, "lifetime_years": 3.0},
    "laptop": {"kwh_per_year": 50.0, "lifetime_years": 5.0},
    "television": {"kwh_per_year": 100.0, "lifetime_years": 8.0},
    "washing_machine": {"kwh_per_year": 200.0, "lifetime_years": 12.0},
    "refrigerator": {"kwh_per_year": 250.0, "lifetime_years": 15.0},
    "led_bulb": {"kwh_per_year": 10.0, "lifetime_years": 15.0},
    "tumble_dryer": {"kwh_per_year": 350.0, "lifetime_years": 12.0},
    "dishwasher": {"kwh_per_year": 260.0, "lifetime_years": 12.0},
}

# End-of-life emission factors (tCO2e per tonne by material)
# Source: DEFRA 2024
EOL_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "plastic": {"landfill": 0.040, "incineration": 2.530, "recycling": -0.500},
    "paper": {"landfill": 1.070, "incineration": 0.021, "recycling": -0.700},
    "glass": {"landfill": 0.009, "incineration": 0.009, "recycling": -0.320},
    "metal_aluminium": {"landfill": 0.009, "incineration": 0.009, "recycling": -9.100},
    "metal_steel": {"landfill": 0.009, "incineration": 0.009, "recycling": -1.800},
    "textiles": {"landfill": 0.470, "incineration": 2.900, "recycling": -3.100},
    "electronics": {"landfill": 0.050, "incineration": 0.100, "recycling": -2.000},
    "organic": {"landfill": 0.580, "composting": -0.050, "anaerobic_digestion": -0.080},
    "wood": {"landfill": 0.830, "incineration": 0.021, "recycling": -0.600},
}

# Retail Scope 3 priority matrix by sub-sector
# CRITICAL = typically >50% of Scope 3; HIGH = >10%; MEDIUM = 1-10%; LOW = <1%
RETAIL_SCOPE3_PRIORITY: Dict[str, Dict[str, str]] = {
    "supermarket": {
        Scope3Category.CAT_1: "CRITICAL",
        Scope3Category.CAT_4: "CRITICAL",
        Scope3Category.CAT_9: "HIGH",
        Scope3Category.CAT_12: "HIGH",
        Scope3Category.CAT_5: "MEDIUM",
        Scope3Category.CAT_7: "MEDIUM",
        Scope3Category.CAT_11: "LOW",
    },
    "fashion_retail": {
        Scope3Category.CAT_1: "CRITICAL",
        Scope3Category.CAT_4: "HIGH",
        Scope3Category.CAT_12: "HIGH",
        Scope3Category.CAT_9: "MEDIUM",
        Scope3Category.CAT_11: "MEDIUM",
        Scope3Category.CAT_5: "MEDIUM",
    },
    "electronics_retail": {
        Scope3Category.CAT_1: "CRITICAL",
        Scope3Category.CAT_11: "CRITICAL",
        Scope3Category.CAT_4: "HIGH",
        Scope3Category.CAT_12: "HIGH",
        Scope3Category.CAT_9: "HIGH",
    },
    "diy_retail": {
        Scope3Category.CAT_1: "CRITICAL",
        Scope3Category.CAT_4: "HIGH",
        Scope3Category.CAT_11: "HIGH",
        Scope3Category.CAT_12: "MEDIUM",
    },
    "general_retail": {
        Scope3Category.CAT_1: "CRITICAL",
        Scope3Category.CAT_4: "HIGH",
        Scope3Category.CAT_9: "MEDIUM",
        Scope3Category.CAT_12: "MEDIUM",
        Scope3Category.CAT_5: "MEDIUM",
    },
}

# Capital goods typical emission factors (tCO2e per EUR million spend)
CAPITAL_GOODS_FACTORS: Dict[str, float] = {
    "store_fitout": 420.0,
    "it_equipment": 580.0,
    "vehicles": 650.0,
    "refrigeration_systems": 720.0,
    "hvac_systems": 550.0,
    "shelving_fixtures": 380.0,
    "security_systems": 310.0,
    "other_capex": 500.0,
}

# WTT (Well-to-Tank) uplift factors for Cat 3
WTT_UPLIFT_FACTORS: Dict[str, float] = {
    "electricity": 0.18,     # 18% WTT uplift on grid electricity
    "natural_gas": 0.20,
    "diesel": 0.24,
    "petrol": 0.22,
}

# Franchise emission intensity (tCO2e per franchise unit per year)
FRANCHISE_INTENSITY: Dict[str, float] = {
    "small_format": 45.0,
    "medium_format": 120.0,
    "large_format": 280.0,
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class PurchasedGoodsData(BaseModel):
    """Purchased goods and services data for Scope 3 Cat 1.

    Attributes:
        product_category: Product category classification.
        sub_category: Optional granular sub-category for better factors.
        spend_eur: Total spend in EUR.
        quantity_units: Quantity in physical units (kg, units, litres).
        quantity_unit_type: Type of physical unit.
        supplier_id: Supplier identifier for tracking.
        supplier_name: Supplier name.
        supplier_emissions_tco2e: Supplier-reported emissions if available.
        calculation_method: Method used for calculation.
        data_quality: Data quality score.
    """
    product_category: ProductCategory
    sub_category: Optional[str] = Field(None, description="Granular sub-category")
    spend_eur: float = Field(0.0, ge=0, description="Spend in EUR")
    quantity_units: Optional[float] = Field(None, ge=0, description="Physical quantity")
    quantity_unit_type: Optional[str] = Field(None, description="Unit type (kg, unit, litre)")
    supplier_id: Optional[str] = Field(None, description="Supplier ID")
    supplier_name: Optional[str] = Field(None, description="Supplier name")
    supplier_emissions_tco2e: Optional[float] = Field(
        None, ge=0, description="Supplier-reported tCO2e"
    )
    calculation_method: CalculationMethod = Field(
        CalculationMethod.SPEND_BASED, description="Calculation method"
    )
    data_quality: DataQualityLevel = Field(
        DataQualityLevel.SCORE_5, description="Data quality"
    )

class CapitalGoodsData(BaseModel):
    """Capital goods data for Scope 3 Cat 2.

    Attributes:
        asset_type: Type of capital asset.
        spend_eur: Total spend in EUR.
        supplier_emissions_tco2e: Supplier-reported if available.
    """
    asset_type: str = Field(..., min_length=1, description="Asset type")
    spend_eur: float = Field(..., ge=0, description="Capital spend (EUR)")
    supplier_emissions_tco2e: Optional[float] = Field(
        None, ge=0, description="Supplier-reported tCO2e"
    )

class TransportData(BaseModel):
    """Transport and distribution data for Scope 3 Cat 4 / Cat 9.

    Attributes:
        mode: Transport mode.
        distance_km: Distance in kilometres.
        weight_tonnes: Cargo weight in tonnes.
        supplier_id: Logistics supplier identifier.
        supplier_emissions_tco2e: Supplier-specific emissions if available.
    """
    mode: TransportMode
    distance_km: float = Field(..., ge=0, description="Distance (km)")
    weight_tonnes: float = Field(..., ge=0, description="Weight (tonnes)")
    supplier_id: Optional[str] = Field(None, description="Logistics supplier")
    supplier_emissions_tco2e: Optional[float] = Field(
        None, ge=0, description="Supplier-reported tCO2e"
    )

class WasteData(BaseModel):
    """Waste generated in operations for Scope 3 Cat 5.

    Attributes:
        waste_type: Type of waste material.
        weight_tonnes: Weight in tonnes.
        disposal_method: Disposal method.
    """
    waste_type: str = Field("mixed", description="Waste type")
    weight_tonnes: float = Field(..., ge=0, description="Weight (tonnes)")
    disposal_method: WasteDisposalMethod = Field(
        WasteDisposalMethod.LANDFILL, description="Disposal method"
    )

class BusinessTravelData(BaseModel):
    """Business travel data for Scope 3 Cat 6.

    Attributes:
        mode: Travel mode.
        distance_km: Distance in km (or nights for hotel).
        passengers: Number of passengers.
    """
    mode: TravelMode
    distance_km: float = Field(..., ge=0, description="Distance (km) or nights")
    passengers: int = Field(1, ge=1, description="Number of travellers")

class CommuteData(BaseModel):
    """Employee commuting data for Scope 3 Cat 7.

    Attributes:
        mode: Commuting mode.
        distance_km_one_way: One-way commute distance.
        employees: Number of employees using this mode.
        working_days_per_year: Working days per year.
    """
    mode: CommuteMode
    distance_km_one_way: float = Field(..., ge=0, description="One-way distance (km)")
    employees: int = Field(..., ge=1, description="Employee count")
    working_days_per_year: int = Field(230, ge=1, le=365, description="Working days/year")

class UsePhaseData(BaseModel):
    """Use-of-sold-products data for Scope 3 Cat 11.

    Attributes:
        product_type: Product type key matching USE_PHASE_ELECTRICITY.
        units_sold: Number of units sold.
        grid_factor_tco2e_per_kwh: Grid emission factor for consumer location.
    """
    product_type: str = Field(..., description="Product type key")
    units_sold: int = Field(..., ge=0, description="Units sold")
    grid_factor_tco2e_per_kwh: float = Field(
        0.000230, ge=0, description="Grid EF (tCO2e/kWh)"
    )

class EndOfLifeData(BaseModel):
    """End-of-life treatment data for Scope 3 Cat 12.

    Attributes:
        material: Material type.
        weight_tonnes: Weight in tonnes.
        disposal_method: End-of-life treatment method.
    """
    material: str = Field(..., description="Material type")
    weight_tonnes: float = Field(..., ge=0, description="Weight (tonnes)")
    disposal_method: str = Field("landfill", description="Disposal method")

class FranchiseData(BaseModel):
    """Franchise data for Scope 3 Cat 14.

    Attributes:
        franchise_format: Size/format of franchise unit.
        unit_count: Number of franchise units.
        reported_emissions_tco2e: Franchisee-reported emissions if available.
    """
    franchise_format: str = Field("medium_format", description="Franchise format")
    unit_count: int = Field(..., ge=1, description="Number of units")
    reported_emissions_tco2e: Optional[float] = Field(
        None, ge=0, description="Reported tCO2e"
    )

class RetailScope3Input(BaseModel):
    """Complete Scope 3 input data for a retail organisation.

    Attributes:
        organisation_id: Organisation identifier.
        reporting_year: Reporting year.
        retail_sub_sector: Sub-sector for priority matrix.
        purchased_goods: Cat 1 data.
        capital_goods: Cat 2 data.
        scope1_tco2e: Scope 1 for Cat 3 WTT calculation.
        scope2_tco2e: Scope 2 for Cat 3 WTT calculation.
        upstream_transport: Cat 4 data.
        waste: Cat 5 data.
        business_travel: Cat 6 data.
        commuting: Cat 7 data.
        upstream_leased_sqm: Cat 8 leased area.
        downstream_transport: Cat 9 data.
        use_phase: Cat 11 data.
        end_of_life: Cat 12 data.
        downstream_leased_sqm: Cat 13 leased area.
        franchises: Cat 14 data.
        investments_eur: Cat 15 invested amount.
        investment_emission_intensity: Cat 15 emission intensity.
    """
    organisation_id: str = Field(..., min_length=1, description="Organisation ID")
    reporting_year: int = Field(..., ge=2020, le=2050, description="Reporting year")
    retail_sub_sector: str = Field("general_retail", description="Retail sub-sector")
    purchased_goods: List[PurchasedGoodsData] = Field(default_factory=list)
    capital_goods: List[CapitalGoodsData] = Field(default_factory=list)
    scope1_tco2e: float = Field(0.0, ge=0, description="Scope 1 for Cat 3")
    scope2_tco2e: float = Field(0.0, ge=0, description="Scope 2 for Cat 3")
    upstream_transport: List[TransportData] = Field(default_factory=list)
    waste: List[WasteData] = Field(default_factory=list)
    business_travel: List[BusinessTravelData] = Field(default_factory=list)
    commuting: List[CommuteData] = Field(default_factory=list)
    upstream_leased_sqm: float = Field(0.0, ge=0, description="Upstream leased m2")
    upstream_leased_ef: float = Field(
        0.080, ge=0, description="Upstream leased EF (tCO2e/m2/yr)"
    )
    downstream_transport: List[TransportData] = Field(default_factory=list)
    use_phase: List[UsePhaseData] = Field(default_factory=list)
    end_of_life: List[EndOfLifeData] = Field(default_factory=list)
    downstream_leased_sqm: float = Field(0.0, ge=0, description="Downstream leased m2")
    downstream_leased_ef: float = Field(
        0.080, ge=0, description="Downstream leased EF (tCO2e/m2/yr)"
    )
    franchises: List[FranchiseData] = Field(default_factory=list)
    investments_eur: float = Field(0.0, ge=0, description="Total investments (EUR)")
    investment_emission_intensity: float = Field(
        0.0003, ge=0, description="Investment EF (tCO2e/EUR)"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class CategoryBreakdown(BaseModel):
    """Breakdown of emissions for a single Scope 3 category.

    Attributes:
        category: Scope 3 category identifier.
        category_name: Human-readable category name.
        emissions_tco2e: Total emissions for this category.
        pct_of_total: Percentage of total Scope 3.
        method_used: Primary calculation method.
        data_quality_score: Weighted data quality score (1-5).
        item_count: Number of input items.
        priority: Priority rating for this retail sub-sector.
    """
    category: str
    category_name: str
    emissions_tco2e: float
    pct_of_total: float
    method_used: str
    data_quality_score: float
    item_count: int
    priority: str

class HotspotResult(BaseModel):
    """Hotspot analysis identifying top emission drivers.

    Attributes:
        top_categories: Top 5 categories by emissions.
        top_suppliers: Top 10 suppliers by emissions.
        top_products: Top 10 product categories by emissions.
        improvement_potential_tco2e: Estimated reduction from method upgrade.
        pareto_categories: Categories comprising 80% of emissions.
    """
    top_categories: List[Dict[str, Any]]
    top_suppliers: List[Dict[str, Any]]
    top_products: List[Dict[str, Any]]
    improvement_potential_tco2e: float
    pareto_categories: List[str]

class DataQualitySummary(BaseModel):
    """Data quality summary across all Scope 3 categories.

    Attributes:
        weighted_score: Emission-weighted data quality score (1=best, 5=worst).
        score_distribution: Count of items by quality score.
        coverage_pct: Percentage of Scope 3 with DQ >= Score 3.
        recommendations: Improvement recommendations.
    """
    weighted_score: float
    score_distribution: Dict[str, int]
    coverage_pct: float
    recommendations: List[str]

class RetailScope3Result(BaseModel):
    """Complete Scope 3 calculation result with provenance.

    Attributes:
        organisation_id: Organisation identifier.
        reporting_year: Reporting year.
        total_scope3_tco2e: Total Scope 3 emissions.
        category_breakdown: Breakdown by category.
        hotspot_analysis: Hotspot identification.
        data_quality_summary: Data quality assessment.
        recommendations: Strategic recommendations.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing time.
        provenance_hash: SHA-256 hash.
    """
    organisation_id: str
    reporting_year: int
    total_scope3_tco2e: float
    category_breakdown: List[CategoryBreakdown]
    hotspot_analysis: HotspotResult
    data_quality_summary: DataQualitySummary
    recommendations: List[str]
    engine_version: str = engine_version
    calculated_at: datetime = Field(default_factory=utcnow)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

# ---------------------------------------------------------------------------
# Category name mapping
# ---------------------------------------------------------------------------

CATEGORY_NAMES: Dict[str, str] = {
    Scope3Category.CAT_1: "Purchased Goods & Services",
    Scope3Category.CAT_2: "Capital Goods",
    Scope3Category.CAT_3: "Fuel- and Energy-Related Activities",
    Scope3Category.CAT_4: "Upstream Transportation & Distribution",
    Scope3Category.CAT_5: "Waste Generated in Operations",
    Scope3Category.CAT_6: "Business Travel",
    Scope3Category.CAT_7: "Employee Commuting",
    Scope3Category.CAT_8: "Upstream Leased Assets",
    Scope3Category.CAT_9: "Downstream Transportation & Distribution",
    Scope3Category.CAT_10: "Processing of Sold Products",
    Scope3Category.CAT_11: "Use of Sold Products",
    Scope3Category.CAT_12: "End-of-Life Treatment of Sold Products",
    Scope3Category.CAT_13: "Downstream Leased Assets",
    Scope3Category.CAT_14: "Franchises",
    Scope3Category.CAT_15: "Investments",
}

# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------

class RetailScope3Engine:
    """Retail-specific Scope 3 emissions calculation engine.

    Covers all 15 GHG Protocol Scope 3 categories with retail-specific
    emission factors, priority matrices, hotspot analysis, and data
    quality scoring. All arithmetic uses Python Decimal.

    Guarantees:
        - Deterministic: identical inputs always produce identical outputs.
        - Reproducible: full provenance via SHA-256 hashing.
        - Auditable: every category calculation is documented.
        - Zero-hallucination: no LLM in the calculation path.

    Usage::

        engine = RetailScope3Engine()
        result = engine.calculate_scope3(input_data)
    """

    def __init__(self) -> None:
        """Initialise the Scope 3 engine with embedded emission factors."""
        self._spend_factors = SPEND_EMISSION_FACTORS
        self._product_factors = PRODUCT_EMISSION_FACTORS
        self._transport_factors = TRANSPORT_EMISSION_FACTORS
        self._waste_factors = WASTE_EMISSION_FACTORS
        self._travel_factors = TRAVEL_EMISSION_FACTORS
        self._commute_factors = COMMUTE_EMISSION_FACTORS
        self._use_phase = USE_PHASE_ELECTRICITY
        self._eol_factors = EOL_EMISSION_FACTORS
        self._capex_factors = CAPITAL_GOODS_FACTORS
        self._priority_matrix = RETAIL_SCOPE3_PRIORITY

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def calculate_scope3(self, input_data: RetailScope3Input) -> RetailScope3Result:
        """Calculate all 15 Scope 3 categories for a retail organisation.

        Args:
            input_data: Complete Scope 3 input data.

        Returns:
            RetailScope3Result with category breakdowns and hotspot analysis.
        """
        t0 = time.perf_counter()

        # Calculate each category
        cat_results: Dict[str, Tuple[Decimal, str, float, int]] = {}

        cat_results[Scope3Category.CAT_1] = self._calc_cat1(input_data.purchased_goods)
        cat_results[Scope3Category.CAT_2] = self._calc_cat2(input_data.capital_goods)
        cat_results[Scope3Category.CAT_3] = self._calc_cat3(
            input_data.scope1_tco2e, input_data.scope2_tco2e
        )
        cat_results[Scope3Category.CAT_4] = self._calc_transport(
            input_data.upstream_transport
        )
        cat_results[Scope3Category.CAT_5] = self._calc_cat5(input_data.waste)
        cat_results[Scope3Category.CAT_6] = self._calc_cat6(input_data.business_travel)
        cat_results[Scope3Category.CAT_7] = self._calc_cat7(input_data.commuting)
        cat_results[Scope3Category.CAT_8] = self._calc_leased(
            input_data.upstream_leased_sqm, input_data.upstream_leased_ef
        )
        cat_results[Scope3Category.CAT_9] = self._calc_transport(
            input_data.downstream_transport
        )
        cat_results[Scope3Category.CAT_10] = (Decimal("0"), "not_applicable", 0.0, 0)
        cat_results[Scope3Category.CAT_11] = self._calc_cat11(input_data.use_phase)
        cat_results[Scope3Category.CAT_12] = self._calc_cat12(input_data.end_of_life)
        cat_results[Scope3Category.CAT_13] = self._calc_leased(
            input_data.downstream_leased_sqm, input_data.downstream_leased_ef
        )
        cat_results[Scope3Category.CAT_14] = self._calc_cat14(input_data.franchises)
        cat_results[Scope3Category.CAT_15] = self._calc_cat15(
            input_data.investments_eur, input_data.investment_emission_intensity
        )

        # Aggregate
        total_scope3 = sum(v[0] for v in cat_results.values())
        priority_map = self._priority_matrix.get(
            input_data.retail_sub_sector,
            self._priority_matrix.get("general_retail", {}),
        )

        # Build category breakdowns
        breakdowns: List[CategoryBreakdown] = []
        for cat_enum in Scope3Category:
            emissions, method, dq, count = cat_results[cat_enum]
            pct = _safe_pct(emissions, total_scope3) if total_scope3 > 0 else Decimal("0")
            priority = priority_map.get(cat_enum, "LOW")
            breakdowns.append(
                CategoryBreakdown(
                    category=cat_enum.value,
                    category_name=CATEGORY_NAMES.get(cat_enum, cat_enum.value),
                    emissions_tco2e=_round_val(emissions, 6),
                    pct_of_total=_round_val(pct, 2),
                    method_used=method,
                    data_quality_score=round(dq, 1),
                    item_count=count,
                    priority=priority,
                )
            )

        # Hotspot analysis
        hotspot = self._build_hotspot(
            breakdowns, input_data.purchased_goods, total_scope3
        )

        # Data quality summary
        dq_summary = self._build_dq_summary(breakdowns, input_data.purchased_goods)

        # Strategic recommendations
        recommendations = self._generate_recommendations(
            breakdowns, input_data.retail_sub_sector
        )

        processing_ms = (time.perf_counter() - t0) * 1000.0

        result = RetailScope3Result(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            total_scope3_tco2e=_round_val(total_scope3, 6),
            category_breakdown=breakdowns,
            hotspot_analysis=hotspot,
            data_quality_summary=dq_summary,
            recommendations=recommendations,
            processing_time_ms=round(processing_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Category calculations
    # -------------------------------------------------------------------

    def _calc_cat1(
        self, goods: List[PurchasedGoodsData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 1: Purchased Goods & Services.

        Supports supplier-specific, product-level, and spend-based methods.

        Args:
            goods: List of purchased goods records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not goods:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        dq_weighted = Decimal("0")
        methods_used = set()

        for g in goods:
            if (
                g.calculation_method == CalculationMethod.SUPPLIER_SPECIFIC
                and g.supplier_emissions_tco2e is not None
            ):
                em = _decimal(g.supplier_emissions_tco2e)
                methods_used.add("supplier_specific")
            elif (
                g.quantity_units is not None
                and g.quantity_units > 0
                and g.sub_category
                and g.sub_category in self._product_factors
            ):
                factor_info = self._product_factors[g.sub_category]
                em = _decimal(g.quantity_units) * _decimal(factor_info["factor"])
                methods_used.add("average_data")
            else:
                # Spend-based fallback
                factor_key = g.sub_category if (
                    g.sub_category and g.sub_category in self._spend_factors
                ) else g.product_category
                factor = _decimal(
                    self._spend_factors.get(factor_key, 600.0)
                )
                spend_millions = _decimal(g.spend_eur) / Decimal("1000000")
                em = spend_millions * factor
                methods_used.add("spend_based")

            total += em
            dq_score = int(g.data_quality.value.split("_")[1])
            dq_weighted += em * _decimal(dq_score)

        avg_dq = float(_safe_divide(dq_weighted, total, Decimal("5")))
        primary_method = "mixed" if len(methods_used) > 1 else methods_used.pop()
        return (total, primary_method, avg_dq, len(goods))

    def _calc_cat2(
        self, capital: List[CapitalGoodsData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 2: Capital Goods.

        Uses supplier-specific data or spend-based EEIO factors.

        Args:
            capital: List of capital goods records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not capital:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        for c in capital:
            if c.supplier_emissions_tco2e is not None:
                total += _decimal(c.supplier_emissions_tco2e)
            else:
                factor = _decimal(
                    self._capex_factors.get(c.asset_type, 500.0)
                )
                spend_millions = _decimal(c.spend_eur) / Decimal("1000000")
                total += spend_millions * factor

        return (total, "spend_based", 4.5, len(capital))

    def _calc_cat3(
        self, scope1: float, scope2: float
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 3: Fuel- and Energy-Related Activities (WTT).

        Applies well-to-tank uplift factors to Scope 1 and Scope 2 totals.

        Args:
            scope1: Total Scope 1 emissions (tCO2e).
            scope2: Total Scope 2 emissions (tCO2e).

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        s1_wtt = _decimal(scope1) * _decimal(
            WTT_UPLIFT_FACTORS.get("natural_gas", 0.20)
        )
        s2_wtt = _decimal(scope2) * _decimal(
            WTT_UPLIFT_FACTORS.get("electricity", 0.18)
        )
        total = s1_wtt + s2_wtt
        count = 1 if (scope1 > 0 or scope2 > 0) else 0
        return (total, "average_data", 3.0, count)

    def _calc_transport(
        self, transport: List[TransportData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate transport emissions (Cat 4 or Cat 9).

        Uses supplier-specific data or tonne-km based emission factors.

        Args:
            transport: List of transport data records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not transport:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        has_supplier = False
        for t in transport:
            if t.supplier_emissions_tco2e is not None:
                total += _decimal(t.supplier_emissions_tco2e)
                has_supplier = True
            else:
                ef = _decimal(self._transport_factors.get(t.mode, 0.000062))
                tkm = _decimal(t.distance_km) * _decimal(t.weight_tonnes)
                total += tkm * ef

        method = "supplier_specific" if has_supplier else "average_data"
        dq = 2.0 if has_supplier else 4.0
        return (total, method, dq, len(transport))

    def _calc_cat5(
        self, waste: List[WasteData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 5: Waste Generated in Operations.

        Uses waste-type and disposal-method specific emission factors.

        Args:
            waste: List of waste data records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not waste:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        for w in waste:
            ef = _decimal(
                self._waste_factors.get(w.disposal_method, 0.586)
            )
            total += _decimal(w.weight_tonnes) * ef

        return (total, "average_data", 3.5, len(waste))

    def _calc_cat6(
        self, travel: List[BusinessTravelData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 6: Business Travel.

        Uses mode-specific emission factors per passenger-km.

        Args:
            travel: List of business travel records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not travel:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        for t in travel:
            ef = _decimal(self._travel_factors.get(t.mode, 0.000131))
            if t.mode == TravelMode.HOTEL:
                # Hotel: distance_km field holds number of nights
                total += _decimal(t.distance_km) * _decimal(t.passengers) * ef
            else:
                total += (
                    _decimal(t.distance_km) * _decimal(t.passengers) * ef
                )

        return (total, "average_data", 3.0, len(travel))

    def _calc_cat7(
        self, commuting: List[CommuteData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 7: Employee Commuting.

        Uses mode-specific factors with return journey and working days.

        Args:
            commuting: List of commuting data records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not commuting:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        for c in commuting:
            ef = _decimal(self._commute_factors.get(c.mode, 0.000171))
            if c.mode == CommuteMode.REMOTE:
                # Remote: per working day, not per km
                total += (
                    _decimal(c.employees)
                    * _decimal(c.working_days_per_year)
                    * ef
                )
            else:
                # Return journey: distance * 2
                annual_km = (
                    _decimal(c.distance_km_one_way)
                    * Decimal("2")
                    * _decimal(c.working_days_per_year)
                )
                total += _decimal(c.employees) * annual_km * ef

        return (total, "average_data", 3.5, len(commuting))

    def _calc_leased(
        self, area_sqm: float, ef_per_sqm: float
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate leased asset emissions (Cat 8 or Cat 13).

        Uses area-based emission intensity factor.

        Args:
            area_sqm: Leased area in square metres.
            ef_per_sqm: Emission factor (tCO2e per m2 per year).

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if area_sqm <= 0:
            return (Decimal("0"), "none", 0.0, 0)

        total = _decimal(area_sqm) * _decimal(ef_per_sqm)
        return (total, "average_data", 4.0, 1)

    def _calc_cat11(
        self, use_phase: List[UsePhaseData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 11: Use of Sold Products.

        Estimates lifetime energy consumption for electrical products.
        Formula: units * kWh/year * lifetime_years * grid_factor

        Args:
            use_phase: List of use-phase data records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not use_phase:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        for u in use_phase:
            product_info = self._use_phase.get(u.product_type)
            if product_info:
                kwh_lifetime = (
                    _decimal(product_info["kwh_per_year"])
                    * _decimal(product_info["lifetime_years"])
                )
                emissions = (
                    _decimal(u.units_sold)
                    * kwh_lifetime
                    * _decimal(u.grid_factor_tco2e_per_kwh)
                )
                total += emissions

        return (total, "average_data", 4.0, len(use_phase))

    def _calc_cat12(
        self, eol: List[EndOfLifeData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 12: End-of-Life Treatment of Sold Products.

        Uses material-specific and disposal-method-specific factors.

        Args:
            eol: List of end-of-life data records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not eol:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        for e in eol:
            material_factors = self._eol_factors.get(e.material, {})
            ef = _decimal(material_factors.get(e.disposal_method, 0.586))
            total += _decimal(e.weight_tonnes) * ef

        return (total, "average_data", 4.0, len(eol))

    def _calc_cat14(
        self, franchises: List[FranchiseData]
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 14: Franchises.

        Uses franchisee-reported data or format-based intensity factors.

        Args:
            franchises: List of franchise data records.

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if not franchises:
            return (Decimal("0"), "none", 0.0, 0)

        total = Decimal("0")
        for f in franchises:
            if f.reported_emissions_tco2e is not None:
                total += _decimal(f.reported_emissions_tco2e)
            else:
                intensity = _decimal(
                    FRANCHISE_INTENSITY.get(f.franchise_format, 120.0)
                )
                total += _decimal(f.unit_count) * intensity

        return (total, "average_data", 4.0, len(franchises))

    def _calc_cat15(
        self, investments_eur: float, intensity: float
    ) -> Tuple[Decimal, str, float, int]:
        """Calculate Cat 15: Investments.

        Uses investment amount and emission intensity factor.

        Args:
            investments_eur: Total investment amount in EUR.
            intensity: Emission intensity factor (tCO2e per EUR).

        Returns:
            Tuple of (emissions, method, dq_score, item_count).
        """
        if investments_eur <= 0:
            return (Decimal("0"), "none", 0.0, 0)

        total = _decimal(investments_eur) * _decimal(intensity)
        return (total, "spend_based", 5.0, 1)

    # -------------------------------------------------------------------
    # Analysis methods
    # -------------------------------------------------------------------

    def _build_hotspot(
        self,
        breakdowns: List[CategoryBreakdown],
        purchased_goods: List[PurchasedGoodsData],
        total_scope3: Decimal,
    ) -> HotspotResult:
        """Build hotspot analysis from category breakdowns.

        Identifies top emission drivers by category, supplier, and product,
        plus Pareto analysis (80/20 rule).

        Args:
            breakdowns: List of category breakdowns.
            purchased_goods: Purchased goods data for supplier/product analysis.
            total_scope3: Total Scope 3 emissions.

        Returns:
            HotspotResult with top drivers and improvement potential.
        """
        # Top categories
        sorted_cats = sorted(
            breakdowns, key=lambda x: x.emissions_tco2e, reverse=True
        )
        top_categories = [
            {
                "category": c.category_name,
                "emissions_tco2e": c.emissions_tco2e,
                "pct": c.pct_of_total,
            }
            for c in sorted_cats[:5]
        ]

        # Top suppliers (from purchased goods)
        supplier_emissions: Dict[str, Decimal] = defaultdict(Decimal)
        product_emissions: Dict[str, Decimal] = defaultdict(Decimal)

        for g in purchased_goods:
            sup_key = g.supplier_name or g.supplier_id or "unknown"
            if g.supplier_emissions_tco2e is not None:
                supplier_emissions[sup_key] += _decimal(g.supplier_emissions_tco2e)
            else:
                factor_key = g.sub_category if (
                    g.sub_category and g.sub_category in self._spend_factors
                ) else g.product_category
                factor = _decimal(self._spend_factors.get(factor_key, 600.0))
                em = _decimal(g.spend_eur) / Decimal("1000000") * factor
                supplier_emissions[sup_key] += em

            product_emissions[g.product_category.value if hasattr(
                g.product_category, 'value'
            ) else str(g.product_category)] += _decimal(g.spend_eur)

        top_suppliers = sorted(
            [
                {"supplier": k, "emissions_tco2e": _round_val(v, 6)}
                for k, v in supplier_emissions.items()
            ],
            key=lambda x: x["emissions_tco2e"],
            reverse=True,
        )[:10]

        top_products = sorted(
            [
                {"product_category": k, "spend_eur": _round_val(v, 2)}
                for k, v in product_emissions.items()
            ],
            key=lambda x: x["spend_eur"],
            reverse=True,
        )[:10]

        # Pareto: categories making up 80% of emissions
        pareto_cats: List[str] = []
        cumulative = Decimal("0")
        threshold = total_scope3 * Decimal("0.80")
        for c in sorted_cats:
            if total_scope3 > 0:
                cumulative += _decimal(c.emissions_tco2e)
                pareto_cats.append(c.category_name)
                if cumulative >= threshold:
                    break

        # Improvement potential: upgrading spend-based to average-data
        improvement = Decimal("0")
        for c in breakdowns:
            if c.method_used == "spend_based" and c.emissions_tco2e > 0:
                improvement += _decimal(c.emissions_tco2e) * Decimal("0.15")

        return HotspotResult(
            top_categories=top_categories,
            top_suppliers=top_suppliers,
            top_products=top_products,
            improvement_potential_tco2e=_round_val(improvement, 6),
            pareto_categories=pareto_cats,
        )

    def _build_dq_summary(
        self,
        breakdowns: List[CategoryBreakdown],
        purchased_goods: List[PurchasedGoodsData],
    ) -> DataQualitySummary:
        """Build data quality summary across all categories.

        Computes emission-weighted data quality score and generates
        improvement recommendations.

        Args:
            breakdowns: List of category breakdowns.
            purchased_goods: Purchased goods data for DQ distribution.

        Returns:
            DataQualitySummary with weighted score and recommendations.
        """
        total_em = Decimal("0")
        weighted_dq = Decimal("0")
        score_dist: Dict[str, int] = {f"score_{i}": 0 for i in range(1, 6)}

        for c in breakdowns:
            if c.emissions_tco2e > 0 and c.data_quality_score > 0:
                em = _decimal(c.emissions_tco2e)
                total_em += em
                weighted_dq += em * _decimal(c.data_quality_score)

        for g in purchased_goods:
            score_dist[g.data_quality.value] = (
                score_dist.get(g.data_quality.value, 0) + 1
            )

        avg_score = float(_safe_divide(weighted_dq, total_em, Decimal("5")))

        # Coverage: % of emissions with DQ <= 3
        good_quality_em = sum(
            _decimal(c.emissions_tco2e)
            for c in breakdowns
            if c.data_quality_score <= 3.0 and c.data_quality_score > 0
        )
        coverage = float(_safe_pct(_decimal(good_quality_em), total_em))

        # Recommendations
        recs: List[str] = []
        if avg_score > 4.0:
            recs.append(
                "Engage top suppliers for primary emissions data to reduce "
                "reliance on spend-based estimates."
            )
        if avg_score > 3.0:
            recs.append(
                "Upgrade calculation methods from spend-based to product-level "
                "average data for high-emission categories."
            )
        if coverage < 50.0:
            recs.append(
                "Less than 50% of Scope 3 emissions have adequate data quality. "
                "Prioritize data collection for Cat 1 and Cat 4."
            )
        spend_count = score_dist.get("score_5", 0)
        total_count = sum(score_dist.values())
        if total_count > 0 and spend_count / max(total_count, 1) > 0.5:
            recs.append(
                "Over 50% of items use spend-based proxies. Consider implementing "
                "supplier engagement programmes for primary data."
            )
        if not recs:
            recs.append(
                "Data quality is good. Continue monitoring and improving "
                "supplier-specific data coverage."
            )

        return DataQualitySummary(
            weighted_score=round(avg_score, 2),
            score_distribution=score_dist,
            coverage_pct=round(coverage, 1),
            recommendations=recs,
        )

    def _generate_recommendations(
        self, breakdowns: List[CategoryBreakdown], sub_sector: str
    ) -> List[str]:
        """Generate strategic recommendations based on results.

        Analyses category breakdown to produce actionable recommendations
        tailored to the retail sub-sector.

        Args:
            breakdowns: Category breakdowns.
            sub_sector: Retail sub-sector.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []
        sorted_cats = sorted(
            breakdowns, key=lambda x: x.emissions_tco2e, reverse=True
        )

        # Top category recommendations
        if sorted_cats and sorted_cats[0].emissions_tco2e > 0:
            top = sorted_cats[0]
            recs.append(
                f"Focus decarbonisation efforts on {top.category_name} "
                f"({top.pct_of_total:.1f}% of Scope 3). This is the highest "
                f"impact area."
            )

        # Category-specific recommendations
        for c in sorted_cats:
            if c.category == Scope3Category.CAT_1 and c.emissions_tco2e > 0:
                recs.append(
                    "Engage top 20% of suppliers (by emissions) for science-based "
                    "targets and primary data sharing via CDP Supply Chain."
                )
                break

        for c in sorted_cats:
            if c.category == Scope3Category.CAT_4 and c.emissions_tco2e > 0:
                recs.append(
                    "Optimise logistics: consolidate shipments, shift road to "
                    "rail/sea where possible, and require GLEC-compliant "
                    "reporting from logistics providers."
                )
                break

        for c in sorted_cats:
            if c.category == Scope3Category.CAT_9 and c.emissions_tco2e > 0:
                recs.append(
                    "For e-commerce deliveries, expand cargo-bike and EV last-mile "
                    "delivery options to reduce downstream transport emissions."
                )
                break

        for c in sorted_cats:
            if c.category == Scope3Category.CAT_12 and c.emissions_tco2e > 0:
                recs.append(
                    "Design for circularity: increase recyclability of packaging "
                    "and products to reduce end-of-life emissions."
                )
                break

        if not recs:
            recs.append(
                "Complete data collection across all Scope 3 categories to "
                "identify priority areas for emissions reduction."
            )

        return recs
