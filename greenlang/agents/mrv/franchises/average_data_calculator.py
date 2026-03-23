# -*- coding: utf-8 -*-
"""
AverageDataCalculatorEngine - AGENT-MRV-027 Engine 3

GHG Protocol Scope 3 Category 14 Tier 2 average-data calculator using
industry EUI benchmarks and revenue intensity factors for franchise unit
emissions estimation.

This engine provides two primary estimation methods when franchise-specific
metered data is unavailable:

1. **Area-Based Benchmark Method**:
   Looks up Energy Use Intensity (EUI) benchmarks by franchise type and
   ASHRAE climate zone, then converts energy to emissions using regional
   grid emission factors:
       E_unit = floor_area_m2 x EUI_benchmark(type, zone) x grid_EF(region)
   Supports 10 franchise types across 5 climate zones.

2. **Revenue-Based Benchmark Method**:
   Applies revenue-based emission intensity factors specific to each
   franchise type:
       E_unit = annual_revenue x revenue_intensity_EF(franchise_type)
   With currency conversion and CPI deflation to base year.

Franchise-Type-Specific Adjustments:
   - QSR: cooking energy overhead (+25-40% for heavy cooking)
   - Hotel: class multiplier (economy -20%, luxury +40%), occupancy rate,
     amenity factors (pool +8%, spa +5%, restaurant +15%)
   - Convenience: 24/7 operation factor (+15%), refrigeration intensity
   - Fitness: equipment power density, shower/sauna usage
   - Automotive: water usage, chemical handling overhead

All calculations use Decimal arithmetic with ROUND_HALF_UP for regulatory
precision. Thread-safe singleton pattern for concurrent pipeline use.

DC-FRN-001: Company-owned units MUST be excluded before passing data to
this engine. This engine calculates emissions only for franchised units.

References:
    - GHG Protocol Technical Guidance for Scope 3, Category 14 (Franchises)
    - US EIA CBECS (Commercial Buildings Energy Consumption Survey)
    - ENERGY STAR Portfolio Manager EUI benchmarks
    - ASHRAE Climate Zone Classification (ASHRAE 169-2021)
    - US EPA eGRID subregional emission factors

Example:
    >>> engine = get_average_data_calculator()
    >>> result = engine.calculate(FranchiseUnitInput(
    ...     unit_id="FRN-001",
    ...     franchise_type=FranchiseType.QSR,
    ...     floor_area_m2=Decimal("250"),
    ...     climate_zone=ClimateZone.ZONE_4A,
    ...     grid_region="US_AVERAGE",
    ...     reporting_year=2024,
    ... ))
    >>> result.total_co2e > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-014
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "average_data_calculator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-014"
AGENT_COMPONENT: str = "AGENT-MRV-027"
TABLE_PREFIX: str = "gl_frn_"

# Decimal precision for rounding (8 decimal places for sub-cent accuracy)
PRECISION: int = 8
ROUNDING: str = ROUND_HALF_UP
_QUANT_8DP: Decimal = Decimal("0.00000001")
_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")

_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")
_HUNDRED: Decimal = Decimal("100")


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class FranchiseType(str, Enum):
    """Franchise type classification for EUI benchmark lookup."""

    QSR = "qsr"                         # Quick-service restaurant (fast food)
    FULL_SERVICE_RESTAURANT = "full_service_restaurant"  # Sit-down restaurant
    HOTEL = "hotel"                     # Hotel / motel / lodging
    CONVENIENCE_STORE = "convenience_store"  # Convenience store / gas station
    RETAIL_CLOTHING = "retail_clothing"  # Retail clothing / apparel franchise
    FITNESS_CENTER = "fitness_center"   # Gym / fitness / recreational center
    AUTOMOTIVE_REPAIR = "automotive_repair"  # Auto repair / service center
    HEALTHCARE_CLINIC = "healthcare_clinic"  # Healthcare / urgent care franchise
    EDUCATION_CENTER = "education_center"  # Tutoring / exam prep / learning center
    COFFEE_SHOP = "coffee_shop"         # Coffee shop / cafe franchise


class ClimateZone(str, Enum):
    """ASHRAE 169-2021 climate zone classification (simplified)."""

    ZONE_1A = "1A"  # Very Hot - Humid (Miami)
    ZONE_2A = "2A"  # Hot - Humid (Houston)
    ZONE_2B = "2B"  # Hot - Dry (Phoenix)
    ZONE_3A = "3A"  # Warm - Humid (Atlanta)
    ZONE_3B = "3B"  # Warm - Dry (Las Vegas)
    ZONE_3C = "3C"  # Warm - Marine (San Francisco)
    ZONE_4A = "4A"  # Mixed - Humid (New York)
    ZONE_4B = "4B"  # Mixed - Dry (Albuquerque)
    ZONE_4C = "4C"  # Mixed - Marine (Seattle)
    ZONE_5A = "5A"  # Cool - Humid (Chicago)
    ZONE_5B = "5B"  # Cool - Dry (Denver)
    ZONE_6A = "6A"  # Cold - Humid (Minneapolis)
    ZONE_6B = "6B"  # Cold - Dry (Helena)
    ZONE_7 = "7"    # Very Cold (Duluth)
    ZONE_8 = "8"    # Subarctic (Fairbanks)


class ClimateZoneGroup(str, Enum):
    """Simplified climate zone groups for EUI benchmark lookup (5 groups)."""

    HOT = "hot"             # Zones 1A, 2A, 2B (Very Hot / Hot)
    WARM = "warm"           # Zones 3A, 3B, 3C (Warm)
    MIXED = "mixed"         # Zones 4A, 4B, 4C (Mixed)
    COOL = "cool"           # Zones 5A, 5B (Cool)
    COLD = "cold"           # Zones 6A, 6B, 7, 8 (Cold / Very Cold / Subarctic)


class HotelClass(str, Enum):
    """Hotel classification affecting energy intensity."""

    ECONOMY = "economy"     # Economy / budget hotel (e.g., Motel 6)
    MIDSCALE = "midscale"   # Midscale hotel (e.g., Holiday Inn Express)
    UPSCALE = "upscale"     # Upscale hotel (e.g., Marriott, Hilton)
    LUXURY = "luxury"       # Luxury hotel (e.g., Ritz-Carlton, Four Seasons)


class CookingIntensity(str, Enum):
    """QSR cooking intensity classification."""

    LIGHT = "light"         # Salads, sandwiches, minimal cooking
    MODERATE = "moderate"   # Standard grill/fryer operation
    HEAVY = "heavy"         # Heavy cooking (pizza ovens, multi-fryer)


class CalculationMethod(str, Enum):
    """Calculation method used for the estimate."""

    AREA_BASED = "area_based"           # EUI benchmark x floor area
    REVENUE_BASED = "revenue_based"     # Revenue x intensity factor
    FRANCHISE_SPECIFIC = "franchise_specific"  # Metered / primary data
    SPEND_BASED = "spend_based"         # EEIO factor x spend


class EFSource(str, Enum):
    """Emission factor data source."""

    EIA_CBECS = "eia_cbecs"         # US EIA CBECS benchmarks
    ENERGY_STAR = "energy_star"     # ENERGY STAR Portfolio Manager
    EPA_EGRID = "epa_egrid"         # US EPA eGRID
    IEA = "iea"                     # IEA country-specific factors
    DEFRA = "defra"                 # UK DEFRA conversion factors
    CUSTOM = "custom"               # Organization-specific factors


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges."""

    TIER_1 = "tier_1"  # Primary metered data (franchise-specific)
    TIER_2 = "tier_2"  # Area-based / revenue-based benchmarks
    TIER_3 = "tier_3"  # Spend-based / generic estimates


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol."""

    REPRESENTATIVENESS = "representativeness"
    COMPLETENESS = "completeness"
    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    TECHNOLOGICAL = "technological"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for revenue-based calculations."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"
    CHF = "CHF"
    SGD = "SGD"
    BRL = "BRL"
    ZAR = "ZAR"
    MXN = "MXN"
    KRW = "KRW"
    NZD = "NZD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    AED = "AED"
    SAR = "SAR"


# ==============================================================================
# REFERENCE DATA TABLES
# ==============================================================================


# EUI Benchmarks: kWh/m2/year by franchise type x climate zone group
# Source: US EIA CBECS, ENERGY STAR Portfolio Manager, industry studies
# 10 franchise types x 5 climate zone groups = 50 data points
FRANCHISE_EUI_BENCHMARKS: Dict[FranchiseType, Dict[ClimateZoneGroup, Decimal]] = {
    FranchiseType.QSR: {
        ClimateZoneGroup.HOT: Decimal("1100.00"),
        ClimateZoneGroup.WARM: Decimal("980.00"),
        ClimateZoneGroup.MIXED: Decimal("920.00"),
        ClimateZoneGroup.COOL: Decimal("960.00"),
        ClimateZoneGroup.COLD: Decimal("1050.00"),
    },
    FranchiseType.FULL_SERVICE_RESTAURANT: {
        ClimateZoneGroup.HOT: Decimal("850.00"),
        ClimateZoneGroup.WARM: Decimal("780.00"),
        ClimateZoneGroup.MIXED: Decimal("720.00"),
        ClimateZoneGroup.COOL: Decimal("750.00"),
        ClimateZoneGroup.COLD: Decimal("820.00"),
    },
    FranchiseType.HOTEL: {
        ClimateZoneGroup.HOT: Decimal("380.00"),
        ClimateZoneGroup.WARM: Decimal("340.00"),
        ClimateZoneGroup.MIXED: Decimal("310.00"),
        ClimateZoneGroup.COOL: Decimal("330.00"),
        ClimateZoneGroup.COLD: Decimal("370.00"),
    },
    FranchiseType.CONVENIENCE_STORE: {
        ClimateZoneGroup.HOT: Decimal("720.00"),
        ClimateZoneGroup.WARM: Decimal("650.00"),
        ClimateZoneGroup.MIXED: Decimal("600.00"),
        ClimateZoneGroup.COOL: Decimal("630.00"),
        ClimateZoneGroup.COLD: Decimal("690.00"),
    },
    FranchiseType.RETAIL_CLOTHING: {
        ClimateZoneGroup.HOT: Decimal("280.00"),
        ClimateZoneGroup.WARM: Decimal("250.00"),
        ClimateZoneGroup.MIXED: Decimal("230.00"),
        ClimateZoneGroup.COOL: Decimal("245.00"),
        ClimateZoneGroup.COLD: Decimal("275.00"),
    },
    FranchiseType.FITNESS_CENTER: {
        ClimateZoneGroup.HOT: Decimal("520.00"),
        ClimateZoneGroup.WARM: Decimal("470.00"),
        ClimateZoneGroup.MIXED: Decimal("430.00"),
        ClimateZoneGroup.COOL: Decimal("455.00"),
        ClimateZoneGroup.COLD: Decimal("510.00"),
    },
    FranchiseType.AUTOMOTIVE_REPAIR: {
        ClimateZoneGroup.HOT: Decimal("310.00"),
        ClimateZoneGroup.WARM: Decimal("280.00"),
        ClimateZoneGroup.MIXED: Decimal("260.00"),
        ClimateZoneGroup.COOL: Decimal("275.00"),
        ClimateZoneGroup.COLD: Decimal("305.00"),
    },
    FranchiseType.HEALTHCARE_CLINIC: {
        ClimateZoneGroup.HOT: Decimal("450.00"),
        ClimateZoneGroup.WARM: Decimal("410.00"),
        ClimateZoneGroup.MIXED: Decimal("380.00"),
        ClimateZoneGroup.COOL: Decimal("400.00"),
        ClimateZoneGroup.COLD: Decimal("440.00"),
    },
    FranchiseType.EDUCATION_CENTER: {
        ClimateZoneGroup.HOT: Decimal("320.00"),
        ClimateZoneGroup.WARM: Decimal("290.00"),
        ClimateZoneGroup.MIXED: Decimal("265.00"),
        ClimateZoneGroup.COOL: Decimal("280.00"),
        ClimateZoneGroup.COLD: Decimal("310.00"),
    },
    FranchiseType.COFFEE_SHOP: {
        ClimateZoneGroup.HOT: Decimal("780.00"),
        ClimateZoneGroup.WARM: Decimal("700.00"),
        ClimateZoneGroup.MIXED: Decimal("650.00"),
        ClimateZoneGroup.COOL: Decimal("680.00"),
        ClimateZoneGroup.COLD: Decimal("750.00"),
    },
}

# Electricity fraction of total EUI by franchise type
# Remainder assumed to be direct fuel (natural gas) combustion
ELECTRICITY_FRACTION: Dict[FranchiseType, Decimal] = {
    FranchiseType.QSR: Decimal("0.55"),
    FranchiseType.FULL_SERVICE_RESTAURANT: Decimal("0.50"),
    FranchiseType.HOTEL: Decimal("0.65"),
    FranchiseType.CONVENIENCE_STORE: Decimal("0.70"),
    FranchiseType.RETAIL_CLOTHING: Decimal("0.85"),
    FranchiseType.FITNESS_CENTER: Decimal("0.75"),
    FranchiseType.AUTOMOTIVE_REPAIR: Decimal("0.40"),
    FranchiseType.HEALTHCARE_CLINIC: Decimal("0.60"),
    FranchiseType.EDUCATION_CENTER: Decimal("0.80"),
    FranchiseType.COFFEE_SHOP: Decimal("0.55"),
}

# Regional grid emission factors (kgCO2e per kWh electricity)
# Source: EPA eGRID 2022, IEA 2023
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    # US regional
    "US_AVERAGE": Decimal("0.3856"),
    "US_NORTHEAST": Decimal("0.2540"),
    "US_SOUTHEAST": Decimal("0.3890"),
    "US_MIDWEST": Decimal("0.4920"),
    "US_SOUTHWEST": Decimal("0.3560"),
    "US_WEST": Decimal("0.2310"),
    "US_TEXAS": Decimal("0.3750"),
    "US_CALIFORNIA": Decimal("0.2050"),
    "US_FLORIDA": Decimal("0.3980"),
    "US_NEW_YORK": Decimal("0.1870"),
    # eGRID subregions (selected)
    "EGRID_RFCW": Decimal("0.4530"),
    "EGRID_SRMW": Decimal("0.5210"),
    "EGRID_CAMX": Decimal("0.2170"),
    "EGRID_ERCT": Decimal("0.3750"),
    "EGRID_NYUP": Decimal("0.1420"),
    "EGRID_NEWE": Decimal("0.2130"),
    "EGRID_RMPA": Decimal("0.4890"),
    "EGRID_NWPP": Decimal("0.2560"),
    "EGRID_SPSO": Decimal("0.4100"),
    "EGRID_SRSO": Decimal("0.3870"),
    # International
    "CA_AVERAGE": Decimal("0.1200"),
    "CA_ONTARIO": Decimal("0.0300"),
    "CA_ALBERTA": Decimal("0.5400"),
    "CA_QUEBEC": Decimal("0.0020"),
    "CA_BC": Decimal("0.0100"),
    "UK": Decimal("0.2070"),
    "EU_AVERAGE": Decimal("0.2560"),
    "DE": Decimal("0.3380"),
    "FR": Decimal("0.0570"),
    "ES": Decimal("0.1610"),
    "IT": Decimal("0.2570"),
    "NL": Decimal("0.3280"),
    "JP": Decimal("0.4570"),
    "AU": Decimal("0.6560"),
    "CN": Decimal("0.5810"),
    "IN": Decimal("0.7080"),
    "BR": Decimal("0.0740"),
    "MX": Decimal("0.4050"),
    "KR": Decimal("0.4590"),
    "SG": Decimal("0.4080"),
    "AE": Decimal("0.4190"),
    "SA": Decimal("0.5690"),
    "GLOBAL_AVERAGE": Decimal("0.4360"),
}

# Direct fuel (natural gas) emission factor (kgCO2e per kWh thermal)
# Source: US EPA, IPCC 2006, DEFRA 2024
NATURAL_GAS_EF: Decimal = Decimal("0.1837")

# Revenue-based intensity factors (kgCO2e per USD revenue)
# Source: Industry averages, CDP benchmarks, EEIO cross-validated
FRANCHISE_REVENUE_INTENSITY: Dict[FranchiseType, Decimal] = {
    FranchiseType.QSR: Decimal("0.1850"),
    FranchiseType.FULL_SERVICE_RESTAURANT: Decimal("0.1620"),
    FranchiseType.HOTEL: Decimal("0.0980"),
    FranchiseType.CONVENIENCE_STORE: Decimal("0.0850"),
    FranchiseType.RETAIL_CLOTHING: Decimal("0.0420"),
    FranchiseType.FITNESS_CENTER: Decimal("0.0730"),
    FranchiseType.AUTOMOTIVE_REPAIR: Decimal("0.0560"),
    FranchiseType.HEALTHCARE_CLINIC: Decimal("0.0490"),
    FranchiseType.EDUCATION_CENTER: Decimal("0.0380"),
    FranchiseType.COFFEE_SHOP: Decimal("0.1550"),
}

# Hotel class multipliers (relative to midscale baseline)
HOTEL_CLASS_MULTIPLIERS: Dict[HotelClass, Decimal] = {
    HotelClass.ECONOMY: Decimal("0.80"),     # -20% from midscale
    HotelClass.MIDSCALE: Decimal("1.00"),    # Baseline
    HotelClass.UPSCALE: Decimal("1.20"),     # +20% from midscale
    HotelClass.LUXURY: Decimal("1.40"),      # +40% from midscale
}

# Hotel amenity factors (additive to base emissions)
HOTEL_AMENITY_FACTORS: Dict[str, Decimal] = {
    "pool": Decimal("0.08"),          # Swimming pool: +8%
    "spa": Decimal("0.05"),           # Spa / wellness center: +5%
    "restaurant": Decimal("0.15"),    # On-site restaurant: +15%
    "fitness_center": Decimal("0.03"),  # On-site gym: +3%
    "laundry": Decimal("0.04"),       # On-site laundry service: +4%
    "conference_center": Decimal("0.06"),  # Conference/meeting rooms: +6%
    "parking_garage": Decimal("0.02"),  # Heated parking garage: +2%
}

# QSR cooking adjustment factors
QSR_COOKING_ADJUSTMENTS: Dict[CookingIntensity, Decimal] = {
    CookingIntensity.LIGHT: Decimal("0.25"),    # +25% for light cooking
    CookingIntensity.MODERATE: Decimal("0.32"),  # +32% for moderate cooking
    CookingIntensity.HEAVY: Decimal("0.40"),     # +40% for heavy cooking
}

# Convenience store 24/7 operation factor
CONVENIENCE_24_7_FACTOR: Decimal = Decimal("0.15")  # +15% for 24/7 ops

# Convenience store refrigeration intensity classes
CONVENIENCE_REFRIGERATION_FACTORS: Dict[str, Decimal] = {
    "low": Decimal("0.05"),      # Small cooler section
    "medium": Decimal("0.12"),   # Standard walk-in cooler
    "high": Decimal("0.20"),     # Extensive refrigeration (frozen section)
}

# Fitness center adjustment factors
FITNESS_EQUIPMENT_DENSITY: Dict[str, Decimal] = {
    "low": Decimal("0.00"),      # Yoga studio, light equipment
    "medium": Decimal("0.08"),   # Standard gym equipment
    "high": Decimal("0.15"),     # Heavy equipment, multiple areas
}

FITNESS_SHOWER_SAUNA_FACTOR: Decimal = Decimal("0.10")  # +10% for shower/sauna

# Automotive repair adjustment factors
AUTOMOTIVE_WATER_USAGE_FACTOR: Decimal = Decimal("0.04")   # +4% water heating
AUTOMOTIVE_CHEMICAL_HANDLING_FACTOR: Decimal = Decimal("0.03")  # +3% ventilation

# Currency exchange rates to USD (mid-market, approximate)
CURRENCY_RATES: Dict[CurrencyCode, Decimal] = {
    CurrencyCode.USD: Decimal("1.0"),
    CurrencyCode.EUR: Decimal("1.0850"),
    CurrencyCode.GBP: Decimal("1.2650"),
    CurrencyCode.CAD: Decimal("0.7410"),
    CurrencyCode.AUD: Decimal("0.6520"),
    CurrencyCode.JPY: Decimal("0.006667"),
    CurrencyCode.CNY: Decimal("0.1378"),
    CurrencyCode.INR: Decimal("0.01198"),
    CurrencyCode.CHF: Decimal("1.1280"),
    CurrencyCode.SGD: Decimal("0.7440"),
    CurrencyCode.BRL: Decimal("0.1990"),
    CurrencyCode.ZAR: Decimal("0.05340"),
    CurrencyCode.MXN: Decimal("0.05680"),
    CurrencyCode.KRW: Decimal("0.000741"),
    CurrencyCode.NZD: Decimal("0.6090"),
    CurrencyCode.SEK: Decimal("0.09530"),
    CurrencyCode.NOK: Decimal("0.09280"),
    CurrencyCode.DKK: Decimal("0.1455"),
    CurrencyCode.AED: Decimal("0.2723"),
    CurrencyCode.SAR: Decimal("0.2666"),
}

# CPI deflators (base year 2021 = 1.0)
CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.8490"),
    2016: Decimal("0.8597"),
    2017: Decimal("0.8781"),
    2018: Decimal("0.8997"),
    2019: Decimal("0.9153"),
    2020: Decimal("0.9271"),
    2021: Decimal("1.0000"),
    2022: Decimal("1.0800"),
    2023: Decimal("1.1152"),
    2024: Decimal("1.1490"),
    2025: Decimal("1.1780"),
}

# Climate zone to group mapping
_ZONE_TO_GROUP: Dict[ClimateZone, ClimateZoneGroup] = {
    ClimateZone.ZONE_1A: ClimateZoneGroup.HOT,
    ClimateZone.ZONE_2A: ClimateZoneGroup.HOT,
    ClimateZone.ZONE_2B: ClimateZoneGroup.HOT,
    ClimateZone.ZONE_3A: ClimateZoneGroup.WARM,
    ClimateZone.ZONE_3B: ClimateZoneGroup.WARM,
    ClimateZone.ZONE_3C: ClimateZoneGroup.WARM,
    ClimateZone.ZONE_4A: ClimateZoneGroup.MIXED,
    ClimateZone.ZONE_4B: ClimateZoneGroup.MIXED,
    ClimateZone.ZONE_4C: ClimateZoneGroup.MIXED,
    ClimateZone.ZONE_5A: ClimateZoneGroup.COOL,
    ClimateZone.ZONE_5B: ClimateZoneGroup.COOL,
    ClimateZone.ZONE_6A: ClimateZoneGroup.COLD,
    ClimateZone.ZONE_6B: ClimateZoneGroup.COLD,
    ClimateZone.ZONE_7: ClimateZoneGroup.COLD,
    ClimateZone.ZONE_8: ClimateZoneGroup.COLD,
}

# Climate zone adjustment factors (relative to MIXED baseline = 1.0)
_CLIMATE_ZONE_ADJUSTMENTS: Dict[ClimateZoneGroup, Decimal] = {
    ClimateZoneGroup.HOT: Decimal("1.12"),   # +12% cooling load
    ClimateZoneGroup.WARM: Decimal("1.05"),  # +5% moderate cooling
    ClimateZoneGroup.MIXED: Decimal("1.00"), # Baseline
    ClimateZoneGroup.COOL: Decimal("1.04"),  # +4% heating load
    ClimateZoneGroup.COLD: Decimal("1.10"),  # +10% heating load
}

# DQI dimension weights (sum to 1.0)
DQI_WEIGHTS: Dict[DQIDimension, Decimal] = {
    DQIDimension.REPRESENTATIVENESS: Decimal("0.30"),
    DQIDimension.COMPLETENESS: Decimal("0.25"),
    DQIDimension.TEMPORAL: Decimal("0.15"),
    DQIDimension.GEOGRAPHICAL: Decimal("0.15"),
    DQIDimension.TECHNOLOGICAL: Decimal("0.15"),
}

# Uncertainty ranges for Tier 2 estimates (half-width of 95% CI as fraction)
TIER_2_UNCERTAINTY: Dict[str, Decimal] = {
    "area_based": Decimal("0.30"),       # +/- 30%
    "revenue_based": Decimal("0.40"),    # +/- 40%
}


# ==============================================================================
# INPUT / OUTPUT MODELS
# ==============================================================================


class HotelOperationsInput(BaseModel):
    """Hotel-specific operational parameters for adjustment calculation."""

    hotel_class: HotelClass = Field(
        default=HotelClass.MIDSCALE,
        description="Hotel class (economy, midscale, upscale, luxury)"
    )
    occupancy_rate: Optional[Decimal] = Field(
        default=None, ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Average occupancy rate (0.0-1.0). None uses default 0.65."
    )
    amenities: List[str] = Field(
        default_factory=list,
        description="List of amenity keys: pool, spa, restaurant, fitness_center, "
                    "laundry, conference_center, parking_garage"
    )

    model_config = ConfigDict(frozen=True)


class QSRCookingInput(BaseModel):
    """QSR-specific cooking energy parameters."""

    cooking_intensity: CookingIntensity = Field(
        default=CookingIntensity.MODERATE,
        description="Cooking intensity (light, moderate, heavy)"
    )

    model_config = ConfigDict(frozen=True)


class ConvenienceStoreInput(BaseModel):
    """Convenience store specific operational parameters."""

    is_24_7: bool = Field(
        default=True,
        description="Whether the store operates 24/7"
    )
    refrigeration_intensity: str = Field(
        default="medium",
        description="Refrigeration intensity: low, medium, high"
    )

    model_config = ConfigDict(frozen=True)


class FitnessInput(BaseModel):
    """Fitness center specific operational parameters."""

    equipment_density: str = Field(
        default="medium",
        description="Equipment power density: low, medium, high"
    )
    has_showers_sauna: bool = Field(
        default=True,
        description="Whether the facility has showers/sauna"
    )

    model_config = ConfigDict(frozen=True)


class AutomotiveInput(BaseModel):
    """Automotive repair specific operational parameters."""

    has_car_wash: bool = Field(
        default=False,
        description="Whether the facility has a car wash bay"
    )
    has_paint_booth: bool = Field(
        default=False,
        description="Whether the facility has a paint/spray booth"
    )

    model_config = ConfigDict(frozen=True)


class FranchiseUnitInput(BaseModel):
    """
    Input for a single franchise unit average-data calculation.

    Supports both area-based and revenue-based methods. If floor_area_m2
    is provided, area-based method is used. If annual_revenue is provided,
    revenue-based method is used. If both are provided, the engine defaults
    to area-based (higher accuracy for Tier 2).

    Example:
        >>> unit = FranchiseUnitInput(
        ...     unit_id="FRN-001",
        ...     franchise_type=FranchiseType.QSR,
        ...     floor_area_m2=Decimal("250"),
        ...     climate_zone=ClimateZone.ZONE_4A,
        ...     grid_region="US_AVERAGE",
        ...     reporting_year=2024,
        ... )
    """

    unit_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Unique identifier for the franchise unit"
    )
    franchise_type: FranchiseType = Field(
        ..., description="Type of franchise (QSR, hotel, convenience, etc.)"
    )
    # Area-based inputs
    floor_area_m2: Optional[Decimal] = Field(
        default=None, gt=Decimal("0"),
        description="Total floor area in square metres"
    )
    climate_zone: Optional[ClimateZone] = Field(
        default=None,
        description="ASHRAE climate zone for EUI adjustment"
    )
    grid_region: Optional[str] = Field(
        default=None,
        description="Grid region code for electricity EF (e.g., US_AVERAGE)"
    )
    # Revenue-based inputs
    annual_revenue: Optional[Decimal] = Field(
        default=None, gt=Decimal("0"),
        description="Annual revenue for the unit"
    )
    revenue_currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of the annual revenue"
    )
    # Common fields
    reporting_year: int = Field(
        default=2024, ge=2015, le=2030,
        description="Reporting year for CPI deflation"
    )
    # Type-specific inputs
    hotel_ops: Optional[HotelOperationsInput] = Field(
        default=None,
        description="Hotel-specific operational parameters"
    )
    qsr_cooking: Optional[QSRCookingInput] = Field(
        default=None,
        description="QSR-specific cooking parameters"
    )
    convenience_ops: Optional[ConvenienceStoreInput] = Field(
        default=None,
        description="Convenience store operational parameters"
    )
    fitness_ops: Optional[FitnessInput] = Field(
        default=None,
        description="Fitness center operational parameters"
    )
    automotive_ops: Optional[AutomotiveInput] = Field(
        default=None,
        description="Automotive repair operational parameters"
    )
    # Partial year
    months_operational: int = Field(
        default=12, ge=1, le=12,
        description="Number of months the unit was operational in reporting year"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityScore(BaseModel):
    """Data quality assessment result."""

    overall_score: Decimal = Field(
        ..., description="Weighted composite DQI score (1.0 - 5.0)"
    )
    tier: DataQualityTier = Field(
        ..., description="Data quality tier classification"
    )
    dimensions: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Score per DQI dimension"
    )
    classification: str = Field(
        ..., description="Quality label: Excellent/Good/Fair/Poor/Very Poor"
    )

    model_config = ConfigDict(frozen=True)


class FranchiseCalculationResult(BaseModel):
    """
    Result from a single franchise unit average-data calculation.

    Contains total CO2e, method used, breakdown of electricity vs fuel
    emissions, type-specific adjustments, and provenance hash.
    """

    unit_id: str = Field(..., description="Franchise unit identifier")
    franchise_type: FranchiseType = Field(
        ..., description="Franchise type used"
    )
    method: CalculationMethod = Field(
        ..., description="Calculation method applied"
    )
    # Emissions breakdown
    electricity_co2e: Decimal = Field(
        ..., description="Electricity-related emissions (kgCO2e)"
    )
    fuel_co2e: Decimal = Field(
        ..., description="Direct fuel combustion emissions (kgCO2e)"
    )
    base_co2e: Decimal = Field(
        ..., description="Base emissions before type-specific adjustments (kgCO2e)"
    )
    adjustment_co2e: Decimal = Field(
        ..., description="Type-specific adjustment emissions (kgCO2e)"
    )
    total_co2e: Decimal = Field(
        ..., description="Total emissions including all adjustments (kgCO2e)"
    )
    total_tco2e: Decimal = Field(
        ..., description="Total emissions in metric tonnes CO2e"
    )
    # Method inputs used
    floor_area_m2: Optional[Decimal] = Field(
        default=None, description="Floor area used (area-based only)"
    )
    eui_kwh_m2: Optional[Decimal] = Field(
        default=None, description="EUI benchmark used (kWh/m2/yr)"
    )
    grid_ef: Optional[Decimal] = Field(
        default=None, description="Grid emission factor used (kgCO2e/kWh)"
    )
    annual_revenue_usd: Optional[Decimal] = Field(
        default=None, description="Revenue in USD used (revenue-based only)"
    )
    revenue_intensity_ef: Optional[Decimal] = Field(
        default=None, description="Revenue intensity factor used (kgCO2e/USD)"
    )
    # Quality and provenance
    data_quality: DataQualityScore = Field(
        ..., description="Data quality assessment"
    )
    uncertainty_lower: Decimal = Field(
        ..., description="Lower bound of 95% CI (kgCO2e)"
    )
    uncertainty_upper: Decimal = Field(
        ..., description="Upper bound of 95% CI (kgCO2e)"
    )
    ef_source: EFSource = Field(
        ..., description="Primary emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculation_timestamp: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )
    engine_version: str = Field(
        default=ENGINE_VERSION,
        description="Engine version that produced this result"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _quantize_8dp(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 decimal places with ROUND_HALF_UP."""
    return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def _quantize_4dp(value: Decimal) -> Decimal:
    """Quantize a Decimal to 4 decimal places with ROUND_HALF_UP."""
    return value.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


def _calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Supports Pydantic models, Decimal values, and any stringifiable objects.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, BaseModel):
            hash_input += json.dumps(
                inp.model_dump(mode="json"), sort_keys=True, default=str
            )
        elif isinstance(inp, Decimal):
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
        else:
            hash_input += str(inp)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def _get_climate_zone_group(zone: ClimateZone) -> ClimateZoneGroup:
    """
    Map a detailed ASHRAE climate zone to a simplified zone group.

    Args:
        zone: Detailed ASHRAE climate zone.

    Returns:
        ClimateZoneGroup for EUI benchmark lookup.
    """
    group = _ZONE_TO_GROUP.get(zone)
    if group is None:
        logger.warning(
            "Unknown climate zone '%s', defaulting to MIXED", zone.value
        )
        return ClimateZoneGroup.MIXED
    return group


def _get_dqi_classification(score: Decimal) -> str:
    """
    Classify a composite DQI score into a human-readable label.

    Args:
        score: Composite DQI score (1-5 scale, 5 = best).

    Returns:
        Classification string.
    """
    if score >= Decimal("4.5"):
        return "Excellent"
    elif score >= Decimal("3.5"):
        return "Good"
    elif score >= Decimal("2.5"):
        return "Fair"
    elif score >= Decimal("1.5"):
        return "Poor"
    else:
        return "Very Poor"


# ==============================================================================
# METRICS COLLECTOR STUB
# ==============================================================================


class _MetricsCollectorStub:
    """Minimal metrics stub when full metrics module is not available."""

    def record_calculation(self, **kwargs: Any) -> None:
        """No-op metric recording."""
        pass

    def record_factor_selection(self, **kwargs: Any) -> None:
        """No-op metric recording."""
        pass

    def record_batch(self, **kwargs: Any) -> None:
        """No-op metric recording."""
        pass


def get_metrics_collector() -> Any:
    """
    Get the metrics collector for the Franchises agent.

    Returns the FranchisesMetrics singleton if available, otherwise falls
    back to a no-op stub.

    Returns:
        Metrics collector instance.
    """
    try:
        from greenlang.agents.mrv.franchises.metrics import get_metrics
        return get_metrics()
    except (ImportError, Exception):
        return _MetricsCollectorStub()


# ==============================================================================
# PROVENANCE MANAGER STUB
# ==============================================================================


class _ProvenanceManagerStub:
    """Minimal provenance stub when full provenance module is not available."""

    def start_chain(self) -> str:
        """Return a placeholder chain ID."""
        import uuid
        return str(uuid.uuid4())

    def record_stage(self, chain_id: str, stage: str,
                     input_data: Any, output_data: Any) -> None:
        """No-op provenance recording."""
        pass

    def seal_chain(self, chain_id: str) -> str:
        """Return a placeholder hash."""
        return hashlib.sha256(chain_id.encode("utf-8")).hexdigest()


def get_provenance_manager() -> Any:
    """
    Get the provenance manager for the Franchises agent.

    Returns the FranchisesProvenance singleton if available, otherwise
    falls back to a stub.

    Returns:
        Provenance manager instance.
    """
    try:
        from greenlang.agents.mrv.franchises.provenance import get_provenance_tracker
        return get_provenance_tracker()
    except (ImportError, Exception):
        return _ProvenanceManagerStub()


# ==============================================================================
# AverageDataCalculatorEngine
# ==============================================================================


class AverageDataCalculatorEngine:
    """
    Tier 2 average-data emissions calculator for franchise units.

    Implements area-based and revenue-based benchmark methods for estimating
    GHG Protocol Scope 3 Category 14 (Franchises) emissions when franchise-
    specific metered data is unavailable.

    Calculation Methods:
        1. Area-based: floor_area x EUI_benchmark x (elec_frac x grid_EF +
           fuel_frac x gas_EF) + type adjustments
        2. Revenue-based: revenue_usd x intensity_EF + type adjustments

    Thread Safety:
        Singleton pattern with threading.RLock for concurrent access.
        All state mutation is protected by the lock.

    Data Quality:
        Average-data estimates are Tier 2 quality. The GHG Protocol recommends
        collecting franchise-specific data (Tier 1) for the top 20% of units
        by revenue/size. Tier 2 is acceptable for intermediate data quality.

    DC-FRN-001:
        Company-owned units MUST be filtered out before calling this engine.
        This engine calculates only for franchised (not company-operated) units.

    Attributes:
        _metrics: Prometheus metrics collector
        _provenance: Provenance tracking manager
        _calculation_count: Running count of calculations performed
        _batch_count: Running count of batch operations

    Example:
        >>> engine = AverageDataCalculatorEngine.get_instance()
        >>> result = engine.calculate(FranchiseUnitInput(
        ...     unit_id="FRN-001",
        ...     franchise_type=FranchiseType.QSR,
        ...     floor_area_m2=Decimal("250"),
        ...     climate_zone=ClimateZone.ZONE_4A,
        ...     grid_region="US_AVERAGE",
        ...     reporting_year=2024,
        ... ))
        >>> result.total_co2e > Decimal("0")
        True
    """

    _instance: Optional["AverageDataCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        """Initialize AverageDataCalculatorEngine with metrics and provenance."""
        self._metrics = get_metrics_collector()
        self._provenance = get_provenance_manager()
        self._calculation_count: int = 0
        self._batch_count: int = 0

        logger.info(
            "AverageDataCalculatorEngine initialized: version=%s, agent=%s, "
            "franchise_types=%d, climate_zones=%d, grid_regions=%d",
            ENGINE_VERSION, AGENT_ID,
            len(FRANCHISE_EUI_BENCHMARKS),
            len(ClimateZoneGroup),
            len(GRID_EMISSION_FACTORS),
        )

    @classmethod
    def get_instance(cls) -> "AverageDataCalculatorEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            AverageDataCalculatorEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing only).

        Thread Safety:
            Protected by the class-level RLock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("AverageDataCalculatorEngine singleton reset")

    # ==========================================================================
    # Primary Public Methods
    # ==========================================================================

    def calculate(
        self, unit_input: FranchiseUnitInput
    ) -> FranchiseCalculationResult:
        """
        Calculate emissions for a franchise unit using average-data methods.

        Selects the appropriate method based on available input:
          - If floor_area_m2 is provided: area-based method (preferred)
          - If annual_revenue is provided: revenue-based method
          - If both are provided: area-based takes precedence

        Args:
            unit_input: Validated franchise unit input data.

        Returns:
            FranchiseCalculationResult with emissions, quality, provenance.

        Raises:
            ValueError: If neither floor_area_m2 nor annual_revenue is provided.
            ValueError: If floor_area_m2 provided but climate_zone/grid_region missing.

        Example:
            >>> result = engine.calculate(FranchiseUnitInput(
            ...     unit_id="FRN-001",
            ...     franchise_type=FranchiseType.HOTEL,
            ...     floor_area_m2=Decimal("5000"),
            ...     climate_zone=ClimateZone.ZONE_3A,
            ...     grid_region="US_SOUTHEAST",
            ...     hotel_ops=HotelOperationsInput(
            ...         hotel_class=HotelClass.UPSCALE,
            ...         amenities=["pool", "restaurant"]
            ...     ),
            ...     reporting_year=2024,
            ... ))
        """
        start_time = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.debug(
            "Starting calculation for unit=%s, type=%s",
            unit_input.unit_id, unit_input.franchise_type.value,
        )

        # Validate that at least one method can be applied
        self._validate_minimum_inputs(unit_input)

        # Determine method to use
        if unit_input.floor_area_m2 is not None:
            method = CalculationMethod.AREA_BASED
            result = self._calculate_area_based_result(unit_input, timestamp)
        else:
            method = CalculationMethod.REVENUE_BASED
            result = self._calculate_revenue_based_result(unit_input, timestamp)

        # Record metrics
        duration = time.monotonic() - start_time
        self._record_calculation_metrics(
            unit_input=unit_input,
            method=method,
            co2e=result.total_co2e,
            duration=duration,
        )
        self._calculation_count += 1

        logger.info(
            "Calculation complete: unit=%s, type=%s, method=%s, "
            "total_co2e=%s kgCO2e, duration=%.4fs",
            unit_input.unit_id,
            unit_input.franchise_type.value,
            method.value,
            result.total_co2e,
            duration,
        )

        return result

    def calculate_area_based(
        self, unit_input: FranchiseUnitInput
    ) -> Decimal:
        """
        Calculate emissions using the area-based EUI benchmark method.

        Formula:
            total_energy_kwh = floor_area_m2 x EUI_benchmark(type, zone)
            elec_kwh = total_energy_kwh x electricity_fraction
            fuel_kwh = total_energy_kwh x (1 - electricity_fraction)
            elec_co2e = elec_kwh x grid_EF(region)
            fuel_co2e = fuel_kwh x NATURAL_GAS_EF
            base_co2e = elec_co2e + fuel_co2e
            (Apply partial-year proration if months_operational < 12)

        Args:
            unit_input: Franchise unit input with floor_area_m2 and climate_zone.

        Returns:
            Base emissions in kgCO2e (before type-specific adjustments).

        Raises:
            ValueError: If floor_area_m2, climate_zone, or grid_region is missing.
        """
        self._validate_area_based_inputs(unit_input)

        # Look up EUI benchmark
        zone_group = _get_climate_zone_group(unit_input.climate_zone)
        eui = self._lookup_eui_benchmark(
            unit_input.franchise_type, zone_group
        )

        # Calculate total energy consumption
        total_energy_kwh = _quantize_8dp(
            unit_input.floor_area_m2 * eui
        )

        # Split into electricity and fuel
        elec_fraction = ELECTRICITY_FRACTION.get(
            unit_input.franchise_type, Decimal("0.60")
        )
        fuel_fraction = _ONE - elec_fraction

        elec_kwh = _quantize_8dp(total_energy_kwh * elec_fraction)
        fuel_kwh = _quantize_8dp(total_energy_kwh * fuel_fraction)

        # Look up grid emission factor
        grid_ef = self._lookup_grid_ef(unit_input.grid_region)

        # Calculate emissions
        elec_co2e = _quantize_8dp(elec_kwh * grid_ef)
        fuel_co2e = _quantize_8dp(fuel_kwh * NATURAL_GAS_EF)
        base_co2e = _quantize_8dp(elec_co2e + fuel_co2e)

        # Apply partial-year proration
        if unit_input.months_operational < 12:
            proration = Decimal(str(unit_input.months_operational)) / Decimal("12")
            base_co2e = _quantize_8dp(base_co2e * proration)

        logger.debug(
            "Area-based calc: floor=%s m2, EUI=%s kWh/m2, "
            "elec_frac=%s, grid_ef=%s, base_co2e=%s kgCO2e",
            unit_input.floor_area_m2, eui, elec_fraction,
            grid_ef, base_co2e,
        )

        return base_co2e

    def calculate_revenue_based(
        self, unit_input: FranchiseUnitInput
    ) -> Decimal:
        """
        Calculate emissions using the revenue-based intensity method.

        Formula:
            revenue_usd = annual_revenue x currency_rate / CPI_deflator
            base_co2e = revenue_usd x revenue_intensity_EF(franchise_type)
            (Apply partial-year proration if months_operational < 12)

        Args:
            unit_input: Franchise unit input with annual_revenue.

        Returns:
            Base emissions in kgCO2e (before type-specific adjustments).

        Raises:
            ValueError: If annual_revenue is None.
        """
        if unit_input.annual_revenue is None:
            raise ValueError(
                f"annual_revenue is required for revenue-based method "
                f"(unit={unit_input.unit_id})"
            )

        # Convert currency to USD
        revenue_usd = self._convert_currency(
            unit_input.annual_revenue,
            unit_input.revenue_currency,
            CurrencyCode.USD,
            unit_input.reporting_year,
        )

        # Apply CPI deflation
        revenue_usd = self._apply_cpi_deflation(
            revenue_usd,
            unit_input.reporting_year,
            2021,  # Base year
        )

        # Look up intensity factor
        intensity_ef = FRANCHISE_REVENUE_INTENSITY.get(
            unit_input.franchise_type
        )
        if intensity_ef is None:
            raise ValueError(
                f"No revenue intensity factor for franchise type "
                f"'{unit_input.franchise_type.value}'"
            )

        # Calculate emissions
        base_co2e = _quantize_8dp(revenue_usd * intensity_ef)

        # Apply partial-year proration
        if unit_input.months_operational < 12:
            proration = Decimal(str(unit_input.months_operational)) / Decimal("12")
            base_co2e = _quantize_8dp(base_co2e * proration)

        logger.debug(
            "Revenue-based calc: revenue=%s %s, revenue_usd=%s, "
            "intensity=%s kgCO2e/USD, base_co2e=%s kgCO2e",
            unit_input.annual_revenue, unit_input.revenue_currency.value,
            revenue_usd, intensity_ef, base_co2e,
        )

        return base_co2e

    # ==========================================================================
    # Type-Specific Adjustment Methods
    # ==========================================================================

    def _apply_franchise_type_adjustment(
        self,
        base_emissions: Decimal,
        franchise_type: FranchiseType,
        unit_input: FranchiseUnitInput,
    ) -> Decimal:
        """
        Apply franchise-type-specific emission adjustments.

        Routes to the appropriate adjustment method based on franchise type.
        Returns the additional emissions (positive value) to add to the base.

        Args:
            base_emissions: Base emissions before adjustments (kgCO2e).
            franchise_type: Franchise type for adjustment selection.
            unit_input: Full unit input for type-specific parameters.

        Returns:
            Additional emissions from type-specific adjustments (kgCO2e).
        """
        adjustment = _ZERO

        if franchise_type == FranchiseType.QSR:
            adjustment = self._get_qsr_cooking_adjustment(
                base_emissions, unit_input.qsr_cooking
            )
        elif franchise_type == FranchiseType.FULL_SERVICE_RESTAURANT:
            # Full-service restaurants get moderate cooking overhead
            cooking = QSRCookingInput(cooking_intensity=CookingIntensity.MODERATE)
            adjustment = self._get_qsr_cooking_adjustment(
                base_emissions, cooking
            )
        elif franchise_type == FranchiseType.HOTEL:
            adjustment = self._get_hotel_adjustment(
                base_emissions, unit_input.hotel_ops
            )
        elif franchise_type == FranchiseType.CONVENIENCE_STORE:
            adjustment = self._get_convenience_adjustment(
                base_emissions, unit_input.convenience_ops
            )
        elif franchise_type == FranchiseType.FITNESS_CENTER:
            adjustment = self._get_fitness_adjustment(
                base_emissions, unit_input.fitness_ops
            )
        elif franchise_type == FranchiseType.AUTOMOTIVE_REPAIR:
            adjustment = self._get_automotive_adjustment(
                base_emissions, unit_input.automotive_ops
            )
        elif franchise_type == FranchiseType.COFFEE_SHOP:
            # Coffee shops get light cooking overhead for espresso machines
            cooking = QSRCookingInput(cooking_intensity=CookingIntensity.LIGHT)
            adjustment = self._get_qsr_cooking_adjustment(
                base_emissions, cooking
            )
        # RETAIL_CLOTHING, HEALTHCARE_CLINIC, EDUCATION_CENTER: no special adjustment

        if adjustment > _ZERO:
            logger.debug(
                "Type adjustment for %s: +%s kgCO2e (%.1f%% of base)",
                franchise_type.value, adjustment,
                float(adjustment / base_emissions * _HUNDRED) if base_emissions > _ZERO else 0,
            )

        return _quantize_8dp(adjustment)

    def _get_hotel_adjustment(
        self,
        base_emissions: Decimal,
        hotel_ops: Optional[HotelOperationsInput],
    ) -> Decimal:
        """
        Calculate hotel-specific emission adjustments.

        Adjustments:
            1. Hotel class multiplier (economy -20%, luxury +40%)
            2. Occupancy rate normalization (vs 65% default occupancy)
            3. Amenity factors (pool +8%, spa +5%, restaurant +15%, etc.)

        Args:
            base_emissions: Base emissions for the hotel unit (kgCO2e).
            hotel_ops: Hotel operational parameters (optional).

        Returns:
            Additional emissions from hotel adjustments (kgCO2e).
        """
        if hotel_ops is None:
            hotel_ops = HotelOperationsInput()

        total_adjustment_factor = _ZERO

        # 1. Hotel class multiplier
        class_multiplier = HOTEL_CLASS_MULTIPLIERS.get(
            hotel_ops.hotel_class, _ONE
        )
        # The class multiplier replaces the base, so adjustment is (multiplier - 1)
        class_adjustment = class_multiplier - _ONE
        total_adjustment_factor = total_adjustment_factor + class_adjustment

        # 2. Occupancy rate normalization
        # If occupancy > 65% default, emissions increase proportionally
        default_occupancy = Decimal("0.65")
        actual_occupancy = hotel_ops.occupancy_rate or default_occupancy
        if actual_occupancy > _ZERO and actual_occupancy != default_occupancy:
            occupancy_adjustment = (actual_occupancy / default_occupancy) - _ONE
            total_adjustment_factor = total_adjustment_factor + occupancy_adjustment

        # 3. Amenity factors
        for amenity in hotel_ops.amenities:
            amenity_key = amenity.lower().strip()
            amenity_factor = HOTEL_AMENITY_FACTORS.get(amenity_key, _ZERO)
            total_adjustment_factor = total_adjustment_factor + amenity_factor

        adjustment = _quantize_8dp(base_emissions * total_adjustment_factor)

        logger.debug(
            "Hotel adjustment: class=%s (%.0f%%), occupancy=%s, "
            "amenities=%s, total_factor=%.2f%%, adjustment=%s kgCO2e",
            hotel_ops.hotel_class.value,
            float(class_adjustment * _HUNDRED),
            hotel_ops.occupancy_rate or "default",
            hotel_ops.amenities,
            float(total_adjustment_factor * _HUNDRED),
            adjustment,
        )

        return adjustment

    def _get_qsr_cooking_adjustment(
        self,
        base_emissions: Decimal,
        cooking_input: Optional[QSRCookingInput],
    ) -> Decimal:
        """
        Calculate QSR cooking energy overhead adjustment.

        QSR franchises have significant additional energy consumption from
        commercial cooking equipment (fryers, grills, ovens). This adjustment
        adds 25-40% overhead depending on cooking intensity.

        Args:
            base_emissions: Base emissions for the QSR unit (kgCO2e).
            cooking_input: QSR cooking parameters (optional).

        Returns:
            Additional emissions from cooking overhead (kgCO2e).
        """
        if cooking_input is None:
            cooking_input = QSRCookingInput()

        cooking_factor = QSR_COOKING_ADJUSTMENTS.get(
            cooking_input.cooking_intensity,
            QSR_COOKING_ADJUSTMENTS[CookingIntensity.MODERATE],
        )

        adjustment = _quantize_8dp(base_emissions * cooking_factor)

        logger.debug(
            "QSR cooking adjustment: intensity=%s, factor=+%.0f%%, "
            "adjustment=%s kgCO2e",
            cooking_input.cooking_intensity.value,
            float(cooking_factor * _HUNDRED),
            adjustment,
        )

        return adjustment

    def _get_convenience_adjustment(
        self,
        base_emissions: Decimal,
        convenience_ops: Optional[ConvenienceStoreInput],
    ) -> Decimal:
        """
        Calculate convenience store specific adjustments.

        Adjustments:
            1. 24/7 operation factor (+15% for extended hours)
            2. Refrigeration intensity (5-20% depending on scale)

        Args:
            base_emissions: Base emissions for the store (kgCO2e).
            convenience_ops: Convenience store parameters (optional).

        Returns:
            Additional emissions from convenience adjustments (kgCO2e).
        """
        if convenience_ops is None:
            convenience_ops = ConvenienceStoreInput()

        total_factor = _ZERO

        # 24/7 operation factor
        if convenience_ops.is_24_7:
            total_factor = total_factor + CONVENIENCE_24_7_FACTOR

        # Refrigeration intensity
        ref_key = convenience_ops.refrigeration_intensity.lower().strip()
        ref_factor = CONVENIENCE_REFRIGERATION_FACTORS.get(
            ref_key, CONVENIENCE_REFRIGERATION_FACTORS["medium"]
        )
        total_factor = total_factor + ref_factor

        adjustment = _quantize_8dp(base_emissions * total_factor)

        logger.debug(
            "Convenience adjustment: 24/7=%s, refrigeration=%s, "
            "factor=+%.0f%%, adjustment=%s kgCO2e",
            convenience_ops.is_24_7,
            convenience_ops.refrigeration_intensity,
            float(total_factor * _HUNDRED),
            adjustment,
        )

        return adjustment

    def _get_fitness_adjustment(
        self,
        base_emissions: Decimal,
        fitness_ops: Optional[FitnessInput],
    ) -> Decimal:
        """
        Calculate fitness center specific adjustments.

        Adjustments:
            1. Equipment power density (0-15% based on density)
            2. Shower/sauna usage (+10% for hot water and steam)

        Args:
            base_emissions: Base emissions for the fitness center (kgCO2e).
            fitness_ops: Fitness center parameters (optional).

        Returns:
            Additional emissions from fitness adjustments (kgCO2e).
        """
        if fitness_ops is None:
            fitness_ops = FitnessInput()

        total_factor = _ZERO

        # Equipment density
        density_key = fitness_ops.equipment_density.lower().strip()
        density_factor = FITNESS_EQUIPMENT_DENSITY.get(
            density_key, FITNESS_EQUIPMENT_DENSITY["medium"]
        )
        total_factor = total_factor + density_factor

        # Shower/sauna
        if fitness_ops.has_showers_sauna:
            total_factor = total_factor + FITNESS_SHOWER_SAUNA_FACTOR

        adjustment = _quantize_8dp(base_emissions * total_factor)

        logger.debug(
            "Fitness adjustment: density=%s, showers=%s, "
            "factor=+%.0f%%, adjustment=%s kgCO2e",
            fitness_ops.equipment_density,
            fitness_ops.has_showers_sauna,
            float(total_factor * _HUNDRED),
            adjustment,
        )

        return adjustment

    def _get_automotive_adjustment(
        self,
        base_emissions: Decimal,
        automotive_ops: Optional[AutomotiveInput],
    ) -> Decimal:
        """
        Calculate automotive repair specific adjustments.

        Adjustments:
            1. Water usage for car wash (+4%)
            2. Chemical handling / ventilation for paint booth (+3%)

        Args:
            base_emissions: Base emissions for the auto repair (kgCO2e).
            automotive_ops: Automotive repair parameters (optional).

        Returns:
            Additional emissions from automotive adjustments (kgCO2e).
        """
        if automotive_ops is None:
            automotive_ops = AutomotiveInput()

        total_factor = _ZERO

        if automotive_ops.has_car_wash:
            total_factor = total_factor + AUTOMOTIVE_WATER_USAGE_FACTOR

        if automotive_ops.has_paint_booth:
            total_factor = total_factor + AUTOMOTIVE_CHEMICAL_HANDLING_FACTOR

        adjustment = _quantize_8dp(base_emissions * total_factor)

        logger.debug(
            "Automotive adjustment: car_wash=%s, paint_booth=%s, "
            "factor=+%.0f%%, adjustment=%s kgCO2e",
            automotive_ops.has_car_wash,
            automotive_ops.has_paint_booth,
            float(total_factor * _HUNDRED),
            adjustment,
        )

        return adjustment

    # ==========================================================================
    # Climate and Currency Adjustment Methods
    # ==========================================================================

    def _apply_climate_adjustment(
        self,
        emissions: Decimal,
        climate_zone: Optional[ClimateZone],
    ) -> Decimal:
        """
        Apply climate zone adjustment to emissions.

        Hot and cold zones have higher emissions due to increased HVAC loads.
        The EUI benchmarks already account for typical climate differences,
        but this provides an additional fine-grained adjustment.

        Args:
            emissions: Base emissions before climate adjustment (kgCO2e).
            climate_zone: ASHRAE climate zone.

        Returns:
            Climate-adjusted emissions (kgCO2e).
        """
        if climate_zone is None:
            return emissions

        zone_group = _get_climate_zone_group(climate_zone)
        adjustment_factor = _CLIMATE_ZONE_ADJUSTMENTS.get(
            zone_group, _ONE
        )

        adjusted = _quantize_8dp(emissions * adjustment_factor)

        if adjustment_factor != _ONE:
            logger.debug(
                "Climate adjustment: zone=%s, group=%s, factor=%s, "
                "%s -> %s kgCO2e",
                climate_zone.value, zone_group.value,
                adjustment_factor, emissions, adjusted,
            )

        return adjusted

    def _convert_currency(
        self,
        amount: Decimal,
        from_currency: CurrencyCode,
        to_currency: CurrencyCode,
        year: int,
    ) -> Decimal:
        """
        Convert an amount between currencies using stored exchange rates.

        Both currencies are converted to USD as an intermediary if neither
        is USD.

        Args:
            amount: Amount to convert.
            from_currency: Source currency.
            to_currency: Target currency.
            year: Reporting year (unused, rates are static).

        Returns:
            Converted amount in target currency.

        Raises:
            ValueError: If currency code not found in CURRENCY_RATES.
        """
        if from_currency == to_currency:
            return amount

        from_rate = CURRENCY_RATES.get(from_currency)
        if from_rate is None:
            raise ValueError(
                f"Currency '{from_currency.value}' not found in CURRENCY_RATES"
            )

        to_rate = CURRENCY_RATES.get(to_currency)
        if to_rate is None:
            raise ValueError(
                f"Currency '{to_currency.value}' not found in CURRENCY_RATES"
            )

        # Convert to USD intermediary, then to target
        amount_usd = _quantize_8dp(amount * from_rate)
        if to_currency == CurrencyCode.USD:
            return amount_usd

        converted = _quantize_8dp(amount_usd / to_rate)
        return converted

    def _apply_cpi_deflation(
        self,
        amount: Decimal,
        from_year: int,
        to_year: int,
    ) -> Decimal:
        """
        Apply CPI deflation to normalize monetary amounts across years.

        Converts nominal spend to real (base-year) USD:
            real_usd = nominal_usd x (base_deflator / year_deflator)

        Args:
            amount: Nominal amount in USD.
            from_year: Year of the spend data.
            to_year: Base year for deflation (typically 2021).

        Returns:
            Deflated amount in base-year USD.

        Raises:
            ValueError: If year not found in CPI_DEFLATORS.
        """
        if from_year == to_year:
            return amount

        from_deflator = CPI_DEFLATORS.get(from_year)
        if from_deflator is None:
            raise ValueError(
                f"CPI deflator not available for year {from_year}. "
                f"Available: {sorted(CPI_DEFLATORS.keys())}"
            )

        to_deflator = CPI_DEFLATORS.get(to_year)
        if to_deflator is None:
            raise ValueError(
                f"CPI deflator not available for year {to_year}. "
                f"Available: {sorted(CPI_DEFLATORS.keys())}"
            )

        deflated = _quantize_8dp(amount * to_deflator / from_deflator)

        logger.debug(
            "CPI deflation: %s (%d) -> %s (%d), deflators %s / %s",
            amount, from_year, deflated, to_year,
            from_deflator, to_deflator,
        )

        return deflated

    # ==========================================================================
    # Data Quality Assessment
    # ==========================================================================

    def _assess_data_quality(
        self,
        unit_input: FranchiseUnitInput,
        method: CalculationMethod,
    ) -> DataQualityScore:
        """
        Assess data quality across 5 GHG Protocol DQI dimensions.

        Scores are on a 1-5 scale (5 = best). Tier 2 average-data methods
        typically score 2-3 on most dimensions.

        Args:
            unit_input: Franchise unit input data.
            method: Calculation method used.

        Returns:
            DataQualityScore with dimension scores and overall assessment.
        """
        dimensions: Dict[str, Decimal] = {}

        # Representativeness: How well does the benchmark represent this unit?
        if method == CalculationMethod.AREA_BASED:
            dimensions[DQIDimension.REPRESENTATIVENESS.value] = Decimal("3.0")
        else:
            dimensions[DQIDimension.REPRESENTATIVENESS.value] = Decimal("2.0")

        # Completeness: How complete is the input data?
        completeness = Decimal("2.0")
        if unit_input.floor_area_m2 is not None:
            completeness = completeness + Decimal("0.5")
        if unit_input.annual_revenue is not None:
            completeness = completeness + Decimal("0.5")
        if unit_input.climate_zone is not None:
            completeness = completeness + Decimal("0.5")
        completeness = min(completeness, Decimal("5.0"))
        dimensions[DQIDimension.COMPLETENESS.value] = completeness

        # Temporal: Is the benchmark current?
        if unit_input.reporting_year >= 2023:
            dimensions[DQIDimension.TEMPORAL.value] = Decimal("4.0")
        elif unit_input.reporting_year >= 2021:
            dimensions[DQIDimension.TEMPORAL.value] = Decimal("3.0")
        else:
            dimensions[DQIDimension.TEMPORAL.value] = Decimal("2.0")

        # Geographical: Does the benchmark match the unit's geography?
        if unit_input.grid_region is not None:
            if unit_input.grid_region.startswith("EGRID_"):
                dimensions[DQIDimension.GEOGRAPHICAL.value] = Decimal("4.0")
            elif unit_input.grid_region.startswith("US_") or "_" in unit_input.grid_region:
                dimensions[DQIDimension.GEOGRAPHICAL.value] = Decimal("3.0")
            else:
                dimensions[DQIDimension.GEOGRAPHICAL.value] = Decimal("3.0")
        else:
            dimensions[DQIDimension.GEOGRAPHICAL.value] = Decimal("1.5")

        # Technological: Does the benchmark match the unit's technology?
        # Type-specific inputs improve this score
        tech_score = Decimal("2.5")
        if unit_input.hotel_ops is not None:
            tech_score = tech_score + Decimal("0.5")
        if unit_input.qsr_cooking is not None:
            tech_score = tech_score + Decimal("0.5")
        if unit_input.convenience_ops is not None:
            tech_score = tech_score + Decimal("0.5")
        if unit_input.fitness_ops is not None:
            tech_score = tech_score + Decimal("0.5")
        if unit_input.automotive_ops is not None:
            tech_score = tech_score + Decimal("0.5")
        tech_score = min(tech_score, Decimal("5.0"))
        dimensions[DQIDimension.TECHNOLOGICAL.value] = tech_score

        # Calculate weighted overall score
        overall = _ZERO
        for dim_name, dim_score in dimensions.items():
            dim_enum = DQIDimension(dim_name)
            weight = DQI_WEIGHTS.get(dim_enum, Decimal("0.20"))
            overall = overall + (dim_score * weight)
        overall = _quantize_4dp(overall)

        # Determine tier
        if overall >= Decimal("4.0"):
            tier = DataQualityTier.TIER_1
        elif overall >= Decimal("2.5"):
            tier = DataQualityTier.TIER_2
        else:
            tier = DataQualityTier.TIER_3

        classification = _get_dqi_classification(overall)

        return DataQualityScore(
            overall_score=overall,
            tier=tier,
            dimensions=dimensions,
            classification=classification,
        )

    # ==========================================================================
    # Batch Processing
    # ==========================================================================

    def calculate_batch(
        self,
        inputs: List[FranchiseUnitInput],
    ) -> List[FranchiseCalculationResult]:
        """
        Calculate emissions for a batch of franchise units.

        Processes each unit sequentially, collecting results and logging
        any per-record errors without aborting the batch.

        Args:
            inputs: List of FranchiseUnitInput records.

        Returns:
            List of FranchiseCalculationResult objects. Failed records
            are excluded and logged at ERROR level.

        Raises:
            ValueError: If inputs list is empty.

        Example:
            >>> results = engine.calculate_batch([unit1, unit2, unit3])
            >>> len(results) <= 3
            True
        """
        if not inputs:
            raise ValueError("Batch inputs list cannot be empty")

        start_time = time.monotonic()
        results: List[FranchiseCalculationResult] = []
        error_count = 0

        logger.info(
            "Starting average-data batch calculation: %d units", len(inputs)
        )

        for idx, unit_input in enumerate(inputs):
            try:
                result = self.calculate(unit_input)
                results.append(result)
            except (ValueError, InvalidOperation) as e:
                error_count += 1
                logger.error(
                    "Batch unit %d failed: %s (unit_id=%s, type=%s)",
                    idx, str(e),
                    unit_input.unit_id,
                    unit_input.franchise_type.value,
                )

        duration = time.monotonic() - start_time
        self._batch_count += 1

        logger.info(
            "Average-data batch complete: %d/%d succeeded, %d failed, "
            "duration=%.4fs",
            len(results), len(inputs), error_count, duration,
        )

        return results

    # ==========================================================================
    # Lookup Helpers
    # ==========================================================================

    def get_available_franchise_types(self) -> List[Dict[str, Any]]:
        """
        Return all supported franchise types with their EUI and intensity data.

        Returns:
            List of dicts with franchise type metadata.
        """
        result = []
        for ftype in FranchiseType:
            eui_data = FRANCHISE_EUI_BENCHMARKS.get(ftype, {})
            intensity = FRANCHISE_REVENUE_INTENSITY.get(ftype, _ZERO)
            elec_frac = ELECTRICITY_FRACTION.get(ftype, Decimal("0.60"))
            result.append({
                "franchise_type": ftype.value,
                "eui_mixed_kwh_m2": float(eui_data.get(ClimateZoneGroup.MIXED, _ZERO)),
                "revenue_intensity_kgco2e_usd": float(intensity),
                "electricity_fraction": float(elec_frac),
            })
        return result

    def get_available_grid_regions(self) -> List[Dict[str, Any]]:
        """
        Return all supported grid regions with emission factors.

        Returns:
            List of dicts with region code and grid EF.
        """
        return [
            {"region": region, "grid_ef_kgco2e_kwh": float(ef)}
            for region, ef in sorted(GRID_EMISSION_FACTORS.items())
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Return operational statistics for this engine.

        Returns:
            Dictionary with calculation counts and configuration info.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "franchise_types": len(FranchiseType),
            "climate_zones": len(ClimateZone),
            "grid_regions": len(GRID_EMISSION_FACTORS),
        }

    # ==========================================================================
    # Private Helpers
    # ==========================================================================

    def _validate_minimum_inputs(
        self, unit_input: FranchiseUnitInput
    ) -> None:
        """Validate that at least one calculation method can be applied."""
        if (unit_input.floor_area_m2 is None
                and unit_input.annual_revenue is None):
            raise ValueError(
                f"At least one of floor_area_m2 or annual_revenue must be "
                f"provided for unit '{unit_input.unit_id}'"
            )

    def _validate_area_based_inputs(
        self, unit_input: FranchiseUnitInput
    ) -> None:
        """Validate area-based method required fields."""
        if unit_input.floor_area_m2 is None:
            raise ValueError(
                f"floor_area_m2 is required for area-based method "
                f"(unit={unit_input.unit_id})"
            )
        if unit_input.climate_zone is None:
            raise ValueError(
                f"climate_zone is required for area-based method "
                f"(unit={unit_input.unit_id})"
            )
        if unit_input.grid_region is None:
            raise ValueError(
                f"grid_region is required for area-based method "
                f"(unit={unit_input.unit_id})"
            )

    def _lookup_eui_benchmark(
        self,
        franchise_type: FranchiseType,
        zone_group: ClimateZoneGroup,
    ) -> Decimal:
        """
        Look up EUI benchmark for franchise type and climate zone group.

        Args:
            franchise_type: Type of franchise.
            zone_group: Simplified climate zone group.

        Returns:
            EUI value in kWh/m2/year.

        Raises:
            ValueError: If no benchmark found.
        """
        type_benchmarks = FRANCHISE_EUI_BENCHMARKS.get(franchise_type)
        if type_benchmarks is None:
            raise ValueError(
                f"No EUI benchmarks for franchise type "
                f"'{franchise_type.value}'"
            )

        eui = type_benchmarks.get(zone_group)
        if eui is None:
            raise ValueError(
                f"No EUI benchmark for franchise type "
                f"'{franchise_type.value}' in climate zone "
                f"'{zone_group.value}'"
            )

        return eui

    def _lookup_grid_ef(self, grid_region: Optional[str]) -> Decimal:
        """
        Look up grid emission factor by region code.

        Falls back to GLOBAL_AVERAGE if the specified region is not found.

        Args:
            grid_region: Grid region code.

        Returns:
            Grid emission factor in kgCO2e/kWh.
        """
        if grid_region is None:
            logger.warning("No grid region specified, using GLOBAL_AVERAGE")
            return GRID_EMISSION_FACTORS["GLOBAL_AVERAGE"]

        ef = GRID_EMISSION_FACTORS.get(grid_region)
        if ef is None:
            logger.warning(
                "Grid region '%s' not found, using GLOBAL_AVERAGE", grid_region
            )
            return GRID_EMISSION_FACTORS["GLOBAL_AVERAGE"]

        return ef

    def _calculate_area_based_result(
        self,
        unit_input: FranchiseUnitInput,
        timestamp: str,
    ) -> FranchiseCalculationResult:
        """
        Build a full FranchiseCalculationResult for the area-based method.

        Args:
            unit_input: Franchise unit input data.
            timestamp: ISO 8601 calculation timestamp.

        Returns:
            Complete calculation result.
        """
        method = CalculationMethod.AREA_BASED

        # Calculate base emissions from area-based method
        base_co2e = self.calculate_area_based(unit_input)

        # Look up values used for the result
        zone_group = _get_climate_zone_group(unit_input.climate_zone)
        eui = self._lookup_eui_benchmark(unit_input.franchise_type, zone_group)
        grid_ef = self._lookup_grid_ef(unit_input.grid_region)
        elec_frac = ELECTRICITY_FRACTION.get(
            unit_input.franchise_type, Decimal("0.60")
        )

        # Calculate electricity and fuel split for result
        total_energy_kwh = _quantize_8dp(unit_input.floor_area_m2 * eui)
        elec_kwh = _quantize_8dp(total_energy_kwh * elec_frac)
        fuel_kwh = _quantize_8dp(total_energy_kwh * (_ONE - elec_frac))
        elec_co2e = _quantize_8dp(elec_kwh * grid_ef)
        fuel_co2e = _quantize_8dp(fuel_kwh * NATURAL_GAS_EF)

        # Apply partial-year proration to split values
        if unit_input.months_operational < 12:
            proration = Decimal(str(unit_input.months_operational)) / Decimal("12")
            elec_co2e = _quantize_8dp(elec_co2e * proration)
            fuel_co2e = _quantize_8dp(fuel_co2e * proration)

        # Apply type-specific adjustments
        adjustment_co2e = self._apply_franchise_type_adjustment(
            base_co2e, unit_input.franchise_type, unit_input
        )

        total_co2e = _quantize_8dp(base_co2e + adjustment_co2e)
        total_tco2e = _quantize_8dp(total_co2e / Decimal("1000"))

        # Assess data quality
        dq = self._assess_data_quality(unit_input, method)

        # Calculate uncertainty bounds
        unc_half = TIER_2_UNCERTAINTY["area_based"]
        unc_lower = _quantize_8dp(total_co2e * (_ONE - unc_half))
        unc_upper = _quantize_8dp(total_co2e * (_ONE + unc_half))

        # Provenance hash
        provenance_hash = _calculate_provenance_hash(
            unit_input, method.value, eui, grid_ef,
            elec_co2e, fuel_co2e, adjustment_co2e, total_co2e,
        )

        return FranchiseCalculationResult(
            unit_id=unit_input.unit_id,
            franchise_type=unit_input.franchise_type,
            method=method,
            electricity_co2e=elec_co2e,
            fuel_co2e=fuel_co2e,
            base_co2e=base_co2e,
            adjustment_co2e=adjustment_co2e,
            total_co2e=total_co2e,
            total_tco2e=total_tco2e,
            floor_area_m2=unit_input.floor_area_m2,
            eui_kwh_m2=eui,
            grid_ef=grid_ef,
            annual_revenue_usd=None,
            revenue_intensity_ef=None,
            data_quality=dq,
            uncertainty_lower=unc_lower,
            uncertainty_upper=unc_upper,
            ef_source=EFSource.EIA_CBECS,
            provenance_hash=provenance_hash,
            calculation_timestamp=timestamp,
            engine_version=ENGINE_VERSION,
        )

    def _calculate_revenue_based_result(
        self,
        unit_input: FranchiseUnitInput,
        timestamp: str,
    ) -> FranchiseCalculationResult:
        """
        Build a full FranchiseCalculationResult for the revenue-based method.

        Args:
            unit_input: Franchise unit input data.
            timestamp: ISO 8601 calculation timestamp.

        Returns:
            Complete calculation result.
        """
        method = CalculationMethod.REVENUE_BASED

        # Convert and deflate revenue
        revenue_usd = self._convert_currency(
            unit_input.annual_revenue,
            unit_input.revenue_currency,
            CurrencyCode.USD,
            unit_input.reporting_year,
        )
        revenue_usd = self._apply_cpi_deflation(
            revenue_usd, unit_input.reporting_year, 2021
        )

        intensity_ef = FRANCHISE_REVENUE_INTENSITY.get(
            unit_input.franchise_type
        )
        if intensity_ef is None:
            raise ValueError(
                f"No revenue intensity factor for franchise type "
                f"'{unit_input.franchise_type.value}'"
            )

        # Calculate base emissions
        base_co2e = self.calculate_revenue_based(unit_input)

        # For revenue-based, we approximate the electricity/fuel split
        elec_frac = ELECTRICITY_FRACTION.get(
            unit_input.franchise_type, Decimal("0.60")
        )
        elec_co2e = _quantize_8dp(base_co2e * elec_frac)
        fuel_co2e = _quantize_8dp(base_co2e * (_ONE - elec_frac))

        # Apply type-specific adjustments
        adjustment_co2e = self._apply_franchise_type_adjustment(
            base_co2e, unit_input.franchise_type, unit_input
        )

        total_co2e = _quantize_8dp(base_co2e + adjustment_co2e)
        total_tco2e = _quantize_8dp(total_co2e / Decimal("1000"))

        # Assess data quality
        dq = self._assess_data_quality(unit_input, method)

        # Calculate uncertainty bounds
        unc_half = TIER_2_UNCERTAINTY["revenue_based"]
        unc_lower = _quantize_8dp(total_co2e * (_ONE - unc_half))
        unc_upper = _quantize_8dp(total_co2e * (_ONE + unc_half))

        # Provenance hash
        provenance_hash = _calculate_provenance_hash(
            unit_input, method.value, revenue_usd, intensity_ef,
            adjustment_co2e, total_co2e,
        )

        return FranchiseCalculationResult(
            unit_id=unit_input.unit_id,
            franchise_type=unit_input.franchise_type,
            method=method,
            electricity_co2e=elec_co2e,
            fuel_co2e=fuel_co2e,
            base_co2e=base_co2e,
            adjustment_co2e=adjustment_co2e,
            total_co2e=total_co2e,
            total_tco2e=total_tco2e,
            floor_area_m2=None,
            eui_kwh_m2=None,
            grid_ef=None,
            annual_revenue_usd=revenue_usd,
            revenue_intensity_ef=intensity_ef,
            data_quality=dq,
            uncertainty_lower=unc_lower,
            uncertainty_upper=unc_upper,
            ef_source=EFSource.EIA_CBECS,
            provenance_hash=provenance_hash,
            calculation_timestamp=timestamp,
            engine_version=ENGINE_VERSION,
        )

    def _record_calculation_metrics(
        self,
        unit_input: FranchiseUnitInput,
        method: CalculationMethod,
        co2e: Decimal,
        duration: float,
    ) -> None:
        """Record calculation metrics to the metrics collector."""
        try:
            self._metrics.record_calculation(
                method=method.value,
                franchise_type=unit_input.franchise_type.value,
                status="success",
                duration=duration,
                co2e=float(co2e),
            )
        except Exception as e:
            logger.warning("Failed to record metrics: %s", e)


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# ==============================================================================


_engine_instance: Optional[AverageDataCalculatorEngine] = None
_engine_lock: threading.RLock = threading.RLock()


def get_average_data_calculator() -> AverageDataCalculatorEngine:
    """
    Get the singleton AverageDataCalculatorEngine instance.

    Thread-safe accessor for the global engine instance.

    Returns:
        AverageDataCalculatorEngine singleton instance.

    Example:
        >>> engine = get_average_data_calculator()
        >>> result = engine.calculate(unit_input)
    """
    global _engine_instance

    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = AverageDataCalculatorEngine.get_instance()

    return _engine_instance


def reset_average_data_calculator() -> None:
    """
    Reset the singleton engine instance (for testing only).

    Convenience function that resets both the module-level and class-level
    singletons. Should only be called in test teardown.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    AverageDataCalculatorEngine.reset_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Constants
    "ENGINE_ID",
    "ENGINE_VERSION",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "TABLE_PREFIX",
    # Enumerations
    "FranchiseType",
    "ClimateZone",
    "ClimateZoneGroup",
    "HotelClass",
    "CookingIntensity",
    "CalculationMethod",
    "EFSource",
    "DataQualityTier",
    "DQIDimension",
    "CurrencyCode",
    # Reference data
    "FRANCHISE_EUI_BENCHMARKS",
    "ELECTRICITY_FRACTION",
    "GRID_EMISSION_FACTORS",
    "NATURAL_GAS_EF",
    "FRANCHISE_REVENUE_INTENSITY",
    "HOTEL_CLASS_MULTIPLIERS",
    "HOTEL_AMENITY_FACTORS",
    "QSR_COOKING_ADJUSTMENTS",
    "CONVENIENCE_24_7_FACTOR",
    "CONVENIENCE_REFRIGERATION_FACTORS",
    "FITNESS_EQUIPMENT_DENSITY",
    "FITNESS_SHOWER_SAUNA_FACTOR",
    "AUTOMOTIVE_WATER_USAGE_FACTOR",
    "AUTOMOTIVE_CHEMICAL_HANDLING_FACTOR",
    "CURRENCY_RATES",
    "CPI_DEFLATORS",
    "DQI_WEIGHTS",
    "TIER_2_UNCERTAINTY",
    # Input models
    "HotelOperationsInput",
    "QSRCookingInput",
    "ConvenienceStoreInput",
    "FitnessInput",
    "AutomotiveInput",
    "FranchiseUnitInput",
    # Output models
    "DataQualityScore",
    "FranchiseCalculationResult",
    # Engine class
    "AverageDataCalculatorEngine",
    # Module-level accessors
    "get_average_data_calculator",
    "reset_average_data_calculator",
]
