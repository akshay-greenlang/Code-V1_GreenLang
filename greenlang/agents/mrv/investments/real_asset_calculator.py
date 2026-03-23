# -*- coding: utf-8 -*-
"""
RealAssetCalculatorEngine - CRE, mortgages, and motor vehicle loans.

This module implements the RealAssetCalculatorEngine for AGENT-MRV-028
(Investments, GHG Protocol Scope 3 Category 15). It provides thread-safe
singleton calculations for financed emissions from real asset investments
using PCAF methodology.

Supported Real Asset Types:
    - Commercial Real Estate (CRE) -- floor area x EUI x grid EF
    - Residential Mortgages -- floor area x EUI x EPC rating x grid EF
    - Motor Vehicle Loans -- annual distance x vehicle EF or lookup

Calculation Formulas:
    Commercial Real Estate:
        attribution_factor = outstanding_amount / property_value_at_origination
        building_emissions = floor_area_m2 x EUI(type, climate_zone) x grid_EF(location)
        financed_emissions = attribution_factor x building_emissions

    Mortgages:
        attribution_factor = outstanding_loan / property_value_at_origination
        building_emissions = floor_area_m2 x residential_EUI(type, zone) x grid_EF
        financed_emissions = attribution_factor x building_emissions

    Motor Vehicle Loans:
        attribution_factor = outstanding_loan / vehicle_value_at_origination
        vehicle_emissions = annual_distance_km x EF(category, fuel_type)
        financed_emissions = attribution_factor x vehicle_emissions

PCAF Data Quality Hierarchy:
    CRE Score 1: Actual metered energy consumption
    CRE Score 2: Energy performance certificate / GRESB rating
    CRE Score 3: Floor area + EUI benchmark
    CRE Score 4: Average per m2 estimate
    CRE Score 5: Average per asset estimate

    Mortgage Score 1: Actual metered energy data
    Mortgage Score 2: EPC rating
    Mortgage Score 3: Floor area + residential EUI
    Mortgage Score 4: Average per m2
    Mortgage Score 5: Average per property

    Vehicle Score 1: Actual fuel/charge data
    Vehicle Score 2: Make/model/year lookup
    Vehicle Score 3: Category + annual distance
    Vehicle Score 4: Category average
    Vehicle Score 5: Portfolio average

Thread Safety:
    Uses __new__ singleton pattern with threading.RLock.

Example:
    >>> engine = RealAssetCalculatorEngine()
    >>> from decimal import Decimal
    >>> from greenlang.agents.mrv.investments.real_asset_calculator import (
    ...     CREInvestmentInput, PropertyType, ClimateZone
    ... )
    >>> cre = CREInvestmentInput(
    ...     outstanding_amount=Decimal("5000000"),
    ...     property_value_at_origination=Decimal("20000000"),
    ...     floor_area_m2=Decimal("5000"),
    ...     property_type=PropertyType.OFFICE,
    ...     climate_zone=ClimateZone.TEMPERATE,
    ...     grid_region="US_AVERAGE",
    ...     reporting_year=2024,
    ... )
    >>> result = engine.calculate_cre(cre)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-015
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-015"
AGENT_COMPONENT: str = "AGENT-MRV-028"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_inv_"

# ==============================================================================
# CONSTANTS
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_MAX_ATTRIBUTION = Decimal("1")
_ENCODING = "utf-8"

# Conversion: kgCO2e to tCO2e
_KG_TO_TONNES = Decimal("0.001")


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class RealAssetType(str, Enum):
    """Type of real asset for financed emissions calculation."""

    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGE = "mortgage"
    MOTOR_VEHICLE_LOAN = "motor_vehicle_loan"


class PropertyType(str, Enum):
    """Commercial or residential property type."""

    OFFICE = "office"
    RETAIL = "retail"
    INDUSTRIAL = "industrial"
    WAREHOUSE = "warehouse"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    EDUCATION = "education"
    MIXED_USE = "mixed_use"
    DATA_CENTER = "data_center"
    RESIDENTIAL_SINGLE = "residential_single"
    RESIDENTIAL_MULTI = "residential_multi"
    RESIDENTIAL_APARTMENT = "residential_apartment"
    RESIDENTIAL_TOWNHOUSE = "residential_townhouse"
    RESIDENTIAL_DETACHED = "residential_detached"
    OTHER = "other"


class ClimateZone(str, Enum):
    """ASHRAE climate zone classification for EUI benchmarks."""

    TROPICAL = "tropical"  # Zone 1: Hot-humid
    ARID = "arid"  # Zone 2: Hot-dry
    TEMPERATE = "temperate"  # Zone 3-4: Mixed
    CONTINENTAL = "continental"  # Zone 5-6: Cold
    SUBARCTIC = "subarctic"  # Zone 7-8: Very cold
    MEDITERRANEAN = "mediterranean"  # Mild winters, dry summers


class EPCRating(str, Enum):
    """Energy Performance Certificate rating (EU standard)."""

    A = "A"  # Best performance
    B = "B"
    C = "C"
    D = "D"  # Baseline
    E = "E"
    F = "F"
    G = "G"  # Worst performance


class VehicleCategory(str, Enum):
    """Vehicle category for motor vehicle loan emissions."""

    PASSENGER_CAR = "passenger_car"
    LIGHT_COMMERCIAL = "light_commercial"
    HEAVY_COMMERCIAL = "heavy_commercial"
    MOTORCYCLE = "motorcycle"
    ELECTRIC_VEHICLE = "electric_vehicle"
    PLUGIN_HYBRID = "plugin_hybrid"
    HYBRID = "hybrid"
    SUV = "suv"
    VAN = "van"
    BUS = "bus"


class VehicleFuelType(str, Enum):
    """Vehicle fuel type for emissions calculation."""

    PETROL = "petrol"
    DIESEL = "diesel"
    LPG = "lpg"
    CNG = "cng"
    ELECTRIC = "electric"
    HYBRID = "hybrid"
    PLUGIN_HYBRID = "plugin_hybrid"
    HYDROGEN = "hydrogen"


class PCAFDataQuality(int, Enum):
    """PCAF data quality score (1 = best, 5 = worst)."""

    SCORE_1 = 1
    SCORE_2 = 2
    SCORE_3 = 3
    SCORE_4 = 4
    SCORE_5 = 5


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes."""

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
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    HKD = "HKD"
    KRW = "KRW"
    NZD = "NZD"
    MXN = "MXN"
    TRY = "TRY"


# ==============================================================================
# EUI BENCHMARKS (kWh per m2 per year)
# Source: ASHRAE 90.1 / ENERGY STAR Portfolio Manager / CRREM
# ==============================================================================

CRE_EUI_BENCHMARKS: Dict[PropertyType, Dict[ClimateZone, Decimal]] = {
    PropertyType.OFFICE: {
        ClimateZone.TROPICAL: Decimal("250.0"),
        ClimateZone.ARID: Decimal("230.0"),
        ClimateZone.TEMPERATE: Decimal("200.0"),
        ClimateZone.CONTINENTAL: Decimal("220.0"),
        ClimateZone.SUBARCTIC: Decimal("260.0"),
        ClimateZone.MEDITERRANEAN: Decimal("190.0"),
    },
    PropertyType.RETAIL: {
        ClimateZone.TROPICAL: Decimal("280.0"),
        ClimateZone.ARID: Decimal("260.0"),
        ClimateZone.TEMPERATE: Decimal("230.0"),
        ClimateZone.CONTINENTAL: Decimal("250.0"),
        ClimateZone.SUBARCTIC: Decimal("290.0"),
        ClimateZone.MEDITERRANEAN: Decimal("220.0"),
    },
    PropertyType.INDUSTRIAL: {
        ClimateZone.TROPICAL: Decimal("180.0"),
        ClimateZone.ARID: Decimal("170.0"),
        ClimateZone.TEMPERATE: Decimal("150.0"),
        ClimateZone.CONTINENTAL: Decimal("160.0"),
        ClimateZone.SUBARCTIC: Decimal("200.0"),
        ClimateZone.MEDITERRANEAN: Decimal("140.0"),
    },
    PropertyType.WAREHOUSE: {
        ClimateZone.TROPICAL: Decimal("120.0"),
        ClimateZone.ARID: Decimal("110.0"),
        ClimateZone.TEMPERATE: Decimal("95.0"),
        ClimateZone.CONTINENTAL: Decimal("105.0"),
        ClimateZone.SUBARCTIC: Decimal("130.0"),
        ClimateZone.MEDITERRANEAN: Decimal("90.0"),
    },
    PropertyType.HOTEL: {
        ClimateZone.TROPICAL: Decimal("320.0"),
        ClimateZone.ARID: Decimal("300.0"),
        ClimateZone.TEMPERATE: Decimal("270.0"),
        ClimateZone.CONTINENTAL: Decimal("290.0"),
        ClimateZone.SUBARCTIC: Decimal("340.0"),
        ClimateZone.MEDITERRANEAN: Decimal("260.0"),
    },
    PropertyType.HOSPITAL: {
        ClimateZone.TROPICAL: Decimal("450.0"),
        ClimateZone.ARID: Decimal("420.0"),
        ClimateZone.TEMPERATE: Decimal("380.0"),
        ClimateZone.CONTINENTAL: Decimal("400.0"),
        ClimateZone.SUBARCTIC: Decimal("470.0"),
        ClimateZone.MEDITERRANEAN: Decimal("360.0"),
    },
    PropertyType.EDUCATION: {
        ClimateZone.TROPICAL: Decimal("200.0"),
        ClimateZone.ARID: Decimal("190.0"),
        ClimateZone.TEMPERATE: Decimal("170.0"),
        ClimateZone.CONTINENTAL: Decimal("180.0"),
        ClimateZone.SUBARCTIC: Decimal("210.0"),
        ClimateZone.MEDITERRANEAN: Decimal("160.0"),
    },
    PropertyType.MIXED_USE: {
        ClimateZone.TROPICAL: Decimal("240.0"),
        ClimateZone.ARID: Decimal("225.0"),
        ClimateZone.TEMPERATE: Decimal("195.0"),
        ClimateZone.CONTINENTAL: Decimal("210.0"),
        ClimateZone.SUBARCTIC: Decimal("250.0"),
        ClimateZone.MEDITERRANEAN: Decimal("185.0"),
    },
    PropertyType.DATA_CENTER: {
        ClimateZone.TROPICAL: Decimal("1200.0"),
        ClimateZone.ARID: Decimal("1100.0"),
        ClimateZone.TEMPERATE: Decimal("1000.0"),
        ClimateZone.CONTINENTAL: Decimal("1050.0"),
        ClimateZone.SUBARCTIC: Decimal("950.0"),
        ClimateZone.MEDITERRANEAN: Decimal("1000.0"),
    },
}

# Residential EUI benchmarks (kWh per m2 per year)
RESIDENTIAL_EUI_BENCHMARKS: Dict[PropertyType, Dict[ClimateZone, Decimal]] = {
    PropertyType.RESIDENTIAL_SINGLE: {
        ClimateZone.TROPICAL: Decimal("140.0"),
        ClimateZone.ARID: Decimal("130.0"),
        ClimateZone.TEMPERATE: Decimal("120.0"),
        ClimateZone.CONTINENTAL: Decimal("150.0"),
        ClimateZone.SUBARCTIC: Decimal("180.0"),
        ClimateZone.MEDITERRANEAN: Decimal("110.0"),
    },
    PropertyType.RESIDENTIAL_MULTI: {
        ClimateZone.TROPICAL: Decimal("110.0"),
        ClimateZone.ARID: Decimal("105.0"),
        ClimateZone.TEMPERATE: Decimal("95.0"),
        ClimateZone.CONTINENTAL: Decimal("120.0"),
        ClimateZone.SUBARCTIC: Decimal("150.0"),
        ClimateZone.MEDITERRANEAN: Decimal("90.0"),
    },
    PropertyType.RESIDENTIAL_APARTMENT: {
        ClimateZone.TROPICAL: Decimal("100.0"),
        ClimateZone.ARID: Decimal("95.0"),
        ClimateZone.TEMPERATE: Decimal("85.0"),
        ClimateZone.CONTINENTAL: Decimal("110.0"),
        ClimateZone.SUBARCTIC: Decimal("135.0"),
        ClimateZone.MEDITERRANEAN: Decimal("80.0"),
    },
    PropertyType.RESIDENTIAL_TOWNHOUSE: {
        ClimateZone.TROPICAL: Decimal("125.0"),
        ClimateZone.ARID: Decimal("118.0"),
        ClimateZone.TEMPERATE: Decimal("108.0"),
        ClimateZone.CONTINENTAL: Decimal("135.0"),
        ClimateZone.SUBARCTIC: Decimal("165.0"),
        ClimateZone.MEDITERRANEAN: Decimal("100.0"),
    },
    PropertyType.RESIDENTIAL_DETACHED: {
        ClimateZone.TROPICAL: Decimal("155.0"),
        ClimateZone.ARID: Decimal("145.0"),
        ClimateZone.TEMPERATE: Decimal("135.0"),
        ClimateZone.CONTINENTAL: Decimal("165.0"),
        ClimateZone.SUBARCTIC: Decimal("200.0"),
        ClimateZone.MEDITERRANEAN: Decimal("125.0"),
    },
}


# EPC rating multipliers relative to D (baseline = 1.0)
EPC_RATING_MULTIPLIERS: Dict[EPCRating, Decimal] = {
    EPCRating.A: Decimal("0.40"),
    EPCRating.B: Decimal("0.60"),
    EPCRating.C: Decimal("0.80"),
    EPCRating.D: Decimal("1.00"),
    EPCRating.E: Decimal("1.20"),
    EPCRating.F: Decimal("1.50"),
    EPCRating.G: Decimal("2.00"),
}


# Grid emission factors by region (kgCO2e per kWh)
# Source: IEA, eGRID, EU EEA
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    # United States
    "US_AVERAGE": Decimal("0.3856"),
    "US_NORTHEAST": Decimal("0.2740"),
    "US_SOUTHEAST": Decimal("0.4120"),
    "US_MIDWEST": Decimal("0.4850"),
    "US_WEST": Decimal("0.2980"),
    "US_TEXAS": Decimal("0.3950"),
    "US_CALIFORNIA": Decimal("0.2050"),
    # Europe
    "EU_AVERAGE": Decimal("0.2560"),
    "UK": Decimal("0.2070"),
    "GERMANY": Decimal("0.3380"),
    "FRANCE": Decimal("0.0520"),
    "SPAIN": Decimal("0.1870"),
    "ITALY": Decimal("0.2580"),
    "NETHERLANDS": Decimal("0.3280"),
    "SWEDEN": Decimal("0.0130"),
    "NORWAY": Decimal("0.0080"),
    "POLAND": Decimal("0.6580"),
    # Asia-Pacific
    "JAPAN": Decimal("0.4520"),
    "CHINA": Decimal("0.5550"),
    "INDIA": Decimal("0.7080"),
    "SOUTH_KOREA": Decimal("0.4590"),
    "AUSTRALIA": Decimal("0.6560"),
    "NEW_ZEALAND": Decimal("0.0960"),
    "SINGAPORE": Decimal("0.4085"),
    # Americas
    "CANADA": Decimal("0.1200"),
    "BRAZIL": Decimal("0.0740"),
    "MEXICO": Decimal("0.4310"),
    # Middle East / Africa
    "UAE": Decimal("0.4850"),
    "SAUDI_ARABIA": Decimal("0.5830"),
    "SOUTH_AFRICA": Decimal("0.9280"),
    # Global default
    "GLOBAL": Decimal("0.4360"),
}


# Vehicle annual emissions by category (kgCO2e per year)
VEHICLE_ANNUAL_EMISSIONS: Dict[VehicleCategory, Decimal] = {
    VehicleCategory.PASSENGER_CAR: Decimal("4600"),
    VehicleCategory.LIGHT_COMMERCIAL: Decimal("6200"),
    VehicleCategory.HEAVY_COMMERCIAL: Decimal("18500"),
    VehicleCategory.MOTORCYCLE: Decimal("1800"),
    VehicleCategory.ELECTRIC_VEHICLE: Decimal("1200"),
    VehicleCategory.PLUGIN_HYBRID: Decimal("2800"),
    VehicleCategory.HYBRID: Decimal("3400"),
    VehicleCategory.SUV: Decimal("5800"),
    VehicleCategory.VAN: Decimal("5500"),
    VehicleCategory.BUS: Decimal("22000"),
}


# Vehicle emission factors per km (kgCO2e per km) by category and fuel
VEHICLE_EF_PER_KM: Dict[VehicleCategory, Dict[VehicleFuelType, Decimal]] = {
    VehicleCategory.PASSENGER_CAR: {
        VehicleFuelType.PETROL: Decimal("0.1710"),
        VehicleFuelType.DIESEL: Decimal("0.1640"),
        VehicleFuelType.LPG: Decimal("0.1530"),
        VehicleFuelType.CNG: Decimal("0.1480"),
        VehicleFuelType.ELECTRIC: Decimal("0.0530"),
        VehicleFuelType.HYBRID: Decimal("0.1100"),
        VehicleFuelType.PLUGIN_HYBRID: Decimal("0.0750"),
        VehicleFuelType.HYDROGEN: Decimal("0.0650"),
    },
    VehicleCategory.LIGHT_COMMERCIAL: {
        VehicleFuelType.PETROL: Decimal("0.2300"),
        VehicleFuelType.DIESEL: Decimal("0.2200"),
        VehicleFuelType.ELECTRIC: Decimal("0.0710"),
        VehicleFuelType.HYBRID: Decimal("0.1600"),
    },
    VehicleCategory.HEAVY_COMMERCIAL: {
        VehicleFuelType.DIESEL: Decimal("0.6800"),
        VehicleFuelType.CNG: Decimal("0.6200"),
        VehicleFuelType.ELECTRIC: Decimal("0.2100"),
    },
    VehicleCategory.SUV: {
        VehicleFuelType.PETROL: Decimal("0.2150"),
        VehicleFuelType.DIESEL: Decimal("0.2050"),
        VehicleFuelType.ELECTRIC: Decimal("0.0680"),
        VehicleFuelType.HYBRID: Decimal("0.1400"),
        VehicleFuelType.PLUGIN_HYBRID: Decimal("0.0950"),
    },
    VehicleCategory.MOTORCYCLE: {
        VehicleFuelType.PETROL: Decimal("0.1130"),
        VehicleFuelType.ELECTRIC: Decimal("0.0200"),
    },
    VehicleCategory.VAN: {
        VehicleFuelType.PETROL: Decimal("0.2100"),
        VehicleFuelType.DIESEL: Decimal("0.2000"),
        VehicleFuelType.ELECTRIC: Decimal("0.0650"),
    },
    VehicleCategory.ELECTRIC_VEHICLE: {
        VehicleFuelType.ELECTRIC: Decimal("0.0530"),
    },
    VehicleCategory.PLUGIN_HYBRID: {
        VehicleFuelType.PLUGIN_HYBRID: Decimal("0.0750"),
    },
    VehicleCategory.HYBRID: {
        VehicleFuelType.HYBRID: Decimal("0.1100"),
    },
    VehicleCategory.BUS: {
        VehicleFuelType.DIESEL: Decimal("0.8200"),
        VehicleFuelType.CNG: Decimal("0.7500"),
        VehicleFuelType.ELECTRIC: Decimal("0.2800"),
    },
}


# Average annual driving distance by category (km per year)
AVERAGE_ANNUAL_DISTANCE_KM: Dict[VehicleCategory, Decimal] = {
    VehicleCategory.PASSENGER_CAR: Decimal("15000"),
    VehicleCategory.LIGHT_COMMERCIAL: Decimal("20000"),
    VehicleCategory.HEAVY_COMMERCIAL: Decimal("45000"),
    VehicleCategory.MOTORCYCLE: Decimal("8000"),
    VehicleCategory.ELECTRIC_VEHICLE: Decimal("15000"),
    VehicleCategory.PLUGIN_HYBRID: Decimal("15000"),
    VehicleCategory.HYBRID: Decimal("15000"),
    VehicleCategory.SUV: Decimal("16000"),
    VehicleCategory.VAN: Decimal("18000"),
    VehicleCategory.BUS: Decimal("50000"),
}


# PCAF uncertainty ranges by data quality score
PCAF_UNCERTAINTY_RANGES: Dict[PCAFDataQuality, Decimal] = {
    PCAFDataQuality.SCORE_1: Decimal("0.05"),
    PCAFDataQuality.SCORE_2: Decimal("0.15"),
    PCAFDataQuality.SCORE_3: Decimal("0.30"),
    PCAFDataQuality.SCORE_4: Decimal("0.45"),
    PCAFDataQuality.SCORE_5: Decimal("0.60"),
}


# Currency exchange rates to USD
CURRENCY_RATES_TO_USD: Dict[CurrencyCode, Decimal] = {
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
    CurrencyCode.SEK: Decimal("0.09250"),
    CurrencyCode.NOK: Decimal("0.09150"),
    CurrencyCode.DKK: Decimal("0.1450"),
    CurrencyCode.HKD: Decimal("0.1282"),
    CurrencyCode.KRW: Decimal("0.000745"),
    CurrencyCode.NZD: Decimal("0.6050"),
    CurrencyCode.MXN: Decimal("0.05820"),
    CurrencyCode.TRY: Decimal("0.03120"),
}

# Default property emission intensities (kgCO2e per m2 per year) as last-resort
DEFAULT_PROPERTY_INTENSITY: Dict[str, Decimal] = {
    "commercial": Decimal("85.0"),
    "residential": Decimal("45.0"),
    "global": Decimal("65.0"),
}


# ==============================================================================
# INPUT MODELS
# ==============================================================================


class CREInvestmentInput(BaseModel):
    """
    Input for commercial real estate financed emissions calculation.

    Example:
        >>> cre = CREInvestmentInput(
        ...     outstanding_amount=Decimal("5000000"),
        ...     property_value_at_origination=Decimal("20000000"),
        ...     floor_area_m2=Decimal("5000"),
        ...     property_type=PropertyType.OFFICE,
        ...     climate_zone=ClimateZone.TEMPERATE,
        ...     grid_region="US_AVERAGE",
        ...     reporting_year=2024,
        ... )
    """

    outstanding_amount: Decimal = Field(
        ..., gt=0, description="Outstanding loan/investment amount"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD, description="Currency of monetary amounts"
    )
    property_value_at_origination: Decimal = Field(
        ..., gt=0, description="Property value at origination (denominator)"
    )
    property_name: Optional[str] = Field(
        default=None, description="Property name or address"
    )
    property_type: PropertyType = Field(
        default=PropertyType.OFFICE, description="Commercial property type"
    )
    climate_zone: ClimateZone = Field(
        default=ClimateZone.TEMPERATE, description="ASHRAE climate zone"
    )
    grid_region: str = Field(
        default="GLOBAL", description="Grid region for electricity EF"
    )
    floor_area_m2: Optional[Decimal] = Field(
        default=None, gt=0, description="Gross floor area in square metres"
    )
    actual_energy_kwh: Optional[Decimal] = Field(
        default=None, ge=0, description="Actual annual energy consumption (kWh)"
    )
    epc_rating: Optional[EPCRating] = Field(
        default=None, description="Energy Performance Certificate rating"
    )
    gresb_score: Optional[int] = Field(
        default=None, ge=0, le=100, description="GRESB rating (0-100)"
    )
    actual_emissions_co2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Actual building emissions (tCO2e), if metered"
    )
    verified: bool = Field(
        default=False, description="Whether emissions data is verified"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030, description="Reporting year"
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier"
    )

    model_config = ConfigDict(frozen=True)


class MortgageInput(BaseModel):
    """
    Input for residential mortgage financed emissions calculation.

    Example:
        >>> mortgage = MortgageInput(
        ...     outstanding_loan=Decimal("200000"),
        ...     property_value_at_origination=Decimal("350000"),
        ...     floor_area_m2=Decimal("120"),
        ...     property_type=PropertyType.RESIDENTIAL_DETACHED,
        ...     climate_zone=ClimateZone.TEMPERATE,
        ...     grid_region="UK",
        ...     epc_rating=EPCRating.C,
        ...     reporting_year=2024,
        ... )
    """

    outstanding_loan: Decimal = Field(
        ..., gt=0, description="Outstanding mortgage balance"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD, description="Currency"
    )
    property_value_at_origination: Decimal = Field(
        ..., gt=0, description="Property value at mortgage origination"
    )
    property_type: PropertyType = Field(
        default=PropertyType.RESIDENTIAL_SINGLE,
        description="Residential property type",
    )
    climate_zone: ClimateZone = Field(
        default=ClimateZone.TEMPERATE, description="Climate zone"
    )
    grid_region: str = Field(
        default="GLOBAL", description="Grid region for electricity EF"
    )
    floor_area_m2: Optional[Decimal] = Field(
        default=None, gt=0, description="Floor area in m2"
    )
    actual_energy_kwh: Optional[Decimal] = Field(
        default=None, ge=0, description="Actual annual energy (kWh)"
    )
    epc_rating: Optional[EPCRating] = Field(
        default=None, description="EPC rating (A-G)"
    )
    actual_emissions_co2e: Optional[Decimal] = Field(
        default=None, ge=0, description="Actual emissions (tCO2e)"
    )
    verified: bool = Field(
        default=False, description="Whether data is verified"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030, description="Reporting year"
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier"
    )

    model_config = ConfigDict(frozen=True)


class MotorVehicleLoanInput(BaseModel):
    """
    Input for motor vehicle loan financed emissions calculation.

    Example:
        >>> vehicle = MotorVehicleLoanInput(
        ...     outstanding_loan=Decimal("25000"),
        ...     vehicle_value_at_origination=Decimal("35000"),
        ...     vehicle_category=VehicleCategory.PASSENGER_CAR,
        ...     fuel_type=VehicleFuelType.PETROL,
        ...     annual_distance_km=Decimal("15000"),
        ...     reporting_year=2024,
        ... )
    """

    outstanding_loan: Decimal = Field(
        ..., gt=0, description="Outstanding vehicle loan balance"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD, description="Currency"
    )
    vehicle_value_at_origination: Decimal = Field(
        ..., gt=0, description="Vehicle value at loan origination"
    )
    vehicle_category: VehicleCategory = Field(
        default=VehicleCategory.PASSENGER_CAR,
        description="Vehicle category",
    )
    fuel_type: VehicleFuelType = Field(
        default=VehicleFuelType.PETROL,
        description="Vehicle fuel type",
    )
    annual_distance_km: Optional[Decimal] = Field(
        default=None, gt=0, description="Annual distance driven (km)"
    )
    actual_fuel_litres: Optional[Decimal] = Field(
        default=None, gt=0, description="Actual annual fuel consumption (litres)"
    )
    actual_emissions_co2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Actual annual vehicle emissions (kgCO2e)"
    )
    make: Optional[str] = Field(
        default=None, description="Vehicle make"
    )
    model_name: Optional[str] = Field(
        default=None, description="Vehicle model"
    )
    year: Optional[int] = Field(
        default=None, ge=1990, le=2030, description="Vehicle model year"
    )
    verified: bool = Field(
        default=False, description="Whether data is verified"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030, description="Reporting year"
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# RESULT MODEL
# ==============================================================================


class InvestmentCalculationResult(BaseModel):
    """Result from a real asset financed emissions calculation."""

    asset_type: str = Field(
        ..., description="Real asset type (cre, mortgage, vehicle)"
    )
    outstanding_amount: Decimal = Field(
        ..., description="Outstanding amount"
    )
    outstanding_amount_usd: Decimal = Field(
        ..., description="Outstanding amount in USD"
    )
    attribution_factor: Decimal = Field(
        ..., description="Attribution factor"
    )
    denominator_type: str = Field(
        ..., description="Type of denominator used"
    )
    denominator_value: Decimal = Field(
        ..., description="Denominator value"
    )
    asset_emissions_co2e: Decimal = Field(
        ..., description="Total asset emissions (tCO2e)"
    )
    financed_emissions_co2e: Decimal = Field(
        ..., description="Financed emissions (tCO2e)"
    )
    pcaf_quality_score: int = Field(
        ..., ge=1, le=5, description="PCAF score"
    )
    uncertainty_lower_co2e: Decimal = Field(
        ..., description="Lower bound of 95% CI"
    )
    uncertainty_upper_co2e: Decimal = Field(
        ..., description="Upper bound of 95% CI"
    )
    calculation_method: str = Field(
        ..., description="Calculation method used"
    )
    calculation_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific calculation details"
    )
    reporting_year: int = Field(
        ..., description="Reporting year"
    )
    processing_time_ms: Decimal = Field(
        ..., description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HASH UTILITIES
# ==============================================================================


def _serialize_for_hash(obj: Any) -> str:
    """Serialize object to deterministic JSON for hashing."""

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "model_dump"):
            return o.model_dump(mode="json")
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default)


def _compute_hash(*inputs: Any) -> str:
    """Compute SHA-256 hash from variable inputs."""
    combined = ""
    for inp in inputs:
        combined += _serialize_for_hash(inp)
    return hashlib.sha256(combined.encode(_ENCODING)).hexdigest()


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class RealAssetCalculatorEngine:
    """
    Thread-safe singleton engine for real asset financed emissions.

    Implements PCAF methodology for commercial real estate, mortgages,
    and motor vehicle loans. All arithmetic uses Python Decimal with
    ROUND_HALF_UP quantization for regulatory precision.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python arithmetic. No LLM calls are used for any numeric computation.

    Thread Safety:
        Uses __new__ singleton with threading.RLock.

    Example:
        >>> engine = RealAssetCalculatorEngine()
        >>> result = engine.calculate_cre(cre_input)
    """

    _instance: Optional["RealAssetCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "RealAssetCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the real asset calculator engine."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._calculation_count: int = 0
        self._count_lock: threading.RLock = threading.RLock()

        logger.info(
            "RealAssetCalculatorEngine initialized: agent=%s, version=%s",
            AGENT_ID, VERSION,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _increment_count(self) -> int:
        """Increment calculation counter thread-safely."""
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize to 8 decimal places."""
        return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

    def _quantize_2dp(self, value: Decimal) -> Decimal:
        """Quantize to 2 decimal places."""
        return value.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)

    def _convert_to_usd(self, amount: Decimal, currency: CurrencyCode) -> Decimal:
        """Convert amount to USD."""
        rate = CURRENCY_RATES_TO_USD.get(currency)
        if rate is None:
            raise ValueError(f"Currency '{currency.value}' not found")
        return self._quantize(amount * rate)

    def _calculate_attribution_factor(
        self, outstanding: Decimal, denominator: Decimal
    ) -> Decimal:
        """Calculate attribution factor, capped at 1.0."""
        if denominator <= _ZERO:
            raise ValueError(f"Denominator must be positive, got {denominator}")
        raw = outstanding / denominator
        return self._quantize(min(raw, _MAX_ATTRIBUTION))

    def _calculate_uncertainty(
        self, financed: Decimal, quality: PCAFDataQuality
    ) -> Tuple[Decimal, Decimal]:
        """Calculate 95% CI bounds."""
        half_width = PCAF_UNCERTAINTY_RANGES.get(quality, Decimal("0.60"))
        delta = self._quantize(financed * half_width)
        lower = self._quantize(max(financed - delta, _ZERO))
        upper = self._quantize(financed + delta)
        return lower, upper

    # =========================================================================
    # BUILDING EMISSIONS (shared CRE + Mortgage logic)
    # =========================================================================

    def _calculate_building_emissions(
        self,
        floor_area_m2: Decimal,
        property_type: PropertyType,
        climate_zone: ClimateZone,
        grid_region: str,
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Calculate annual building emissions from floor area, EUI, and grid EF.

        Formula:
            energy_kwh = floor_area_m2 x EUI(property_type, climate_zone)
            emissions_kgCO2e = energy_kwh x grid_EF(region)
            emissions_tCO2e = emissions_kgCO2e / 1000

        Args:
            floor_area_m2: Gross floor area in square metres.
            property_type: Property type for EUI lookup.
            climate_zone: Climate zone for EUI lookup.
            grid_region: Grid region for emission factor lookup.

        Returns:
            Tuple of (emissions_tco2e, details_dict).
        """
        # Look up EUI
        eui: Optional[Decimal] = None

        # Try CRE benchmarks first
        if property_type in CRE_EUI_BENCHMARKS:
            zone_map = CRE_EUI_BENCHMARKS[property_type]
            eui = zone_map.get(climate_zone)

        # Try residential benchmarks
        if eui is None and property_type in RESIDENTIAL_EUI_BENCHMARKS:
            zone_map = RESIDENTIAL_EUI_BENCHMARKS[property_type]
            eui = zone_map.get(climate_zone)

        # Fallback to OTHER/default
        if eui is None:
            if property_type in CRE_EUI_BENCHMARKS:
                eui = CRE_EUI_BENCHMARKS[property_type].get(
                    ClimateZone.TEMPERATE, Decimal("200.0")
                )
            else:
                eui = Decimal("120.0")  # Generic residential fallback

        # Calculate energy consumption
        energy_kwh = self._quantize(floor_area_m2 * eui)

        # Look up grid emission factor
        grid_ef = GRID_EMISSION_FACTORS.get(
            grid_region.upper(),
            GRID_EMISSION_FACTORS["GLOBAL"],
        )

        # Calculate emissions
        emissions_kg = self._quantize(energy_kwh * grid_ef)
        emissions_tco2e = self._quantize(emissions_kg * _KG_TO_TONNES)

        details = {
            "floor_area_m2": str(floor_area_m2),
            "eui_kwh_per_m2": str(eui),
            "energy_kwh": str(energy_kwh),
            "grid_region": grid_region,
            "grid_ef_kgco2e_per_kwh": str(grid_ef),
            "emissions_kgco2e": str(emissions_kg),
            "emissions_tco2e": str(emissions_tco2e),
        }

        return emissions_tco2e, details

    def _apply_epc_rating(
        self, base_eui: Decimal, rating: EPCRating
    ) -> Decimal:
        """
        Apply EPC rating multiplier to base EUI.

        EPC ratings adjust the energy intensity relative to baseline (D=1.0):
            A: 0.4x, B: 0.6x, C: 0.8x, D: 1.0x, E: 1.2x, F: 1.5x, G: 2.0x

        Args:
            base_eui: Baseline EUI in kWh/m2/year.
            rating: EPC rating (A-G).

        Returns:
            Adjusted EUI in kWh/m2/year.
        """
        multiplier = EPC_RATING_MULTIPLIERS.get(rating, _ONE)
        return self._quantize(base_eui * multiplier)

    # =========================================================================
    # VEHICLE EMISSIONS
    # =========================================================================

    def _calculate_vehicle_emissions(
        self,
        category: VehicleCategory,
        annual_km: Optional[Decimal],
        fuel_type: VehicleFuelType,
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Calculate annual vehicle emissions.

        Priority:
            1. Distance-based: annual_km x EF(category, fuel_type)
            2. Category average: fixed annual emissions

        Args:
            category: Vehicle category.
            annual_km: Annual distance in km (or None for average).
            fuel_type: Vehicle fuel type.

        Returns:
            Tuple of (emissions_tco2e, details_dict).
        """
        method: str
        emissions_kg: Decimal

        if annual_km is not None:
            # Distance-based calculation
            category_efs = VEHICLE_EF_PER_KM.get(category, {})
            ef = category_efs.get(fuel_type)

            if ef is None:
                # Fallback: try finding any EF for this category
                if category_efs:
                    ef = next(iter(category_efs.values()))
                else:
                    # Ultimate fallback: use annual average
                    annual_avg = VEHICLE_ANNUAL_EMISSIONS.get(
                        category, Decimal("4600")
                    )
                    emissions_tco2e = self._quantize(annual_avg * _KG_TO_TONNES)
                    details = {
                        "method": "category_average_fallback",
                        "category": category.value,
                        "annual_emissions_kgco2e": str(annual_avg),
                    }
                    return emissions_tco2e, details

            emissions_kg = self._quantize(annual_km * ef)
            method = "distance_based"
            details = {
                "method": method,
                "annual_km": str(annual_km),
                "fuel_type": fuel_type.value,
                "ef_kgco2e_per_km": str(ef),
                "emissions_kgco2e": str(emissions_kg),
            }
        else:
            # Category average
            emissions_kg = VEHICLE_ANNUAL_EMISSIONS.get(
                category, Decimal("4600")
            )
            method = "category_average"
            details = {
                "method": method,
                "category": category.value,
                "annual_emissions_kgco2e": str(emissions_kg),
            }

        emissions_tco2e = self._quantize(emissions_kg * _KG_TO_TONNES)
        details["emissions_tco2e"] = str(emissions_tco2e)

        return emissions_tco2e, details

    # =========================================================================
    # PCAF QUALITY DETERMINATION
    # =========================================================================

    def _determine_pcaf_quality_cre(
        self, input_data: CREInvestmentInput
    ) -> PCAFDataQuality:
        """Determine PCAF quality for CRE."""
        if input_data.actual_emissions_co2e is not None and input_data.verified:
            return PCAFDataQuality.SCORE_1
        if input_data.actual_energy_kwh is not None:
            return PCAFDataQuality.SCORE_1
        if input_data.epc_rating is not None or input_data.gresb_score is not None:
            return PCAFDataQuality.SCORE_2
        if input_data.floor_area_m2 is not None:
            return PCAFDataQuality.SCORE_3
        return PCAFDataQuality.SCORE_5

    def _determine_pcaf_quality_mortgage(
        self, input_data: MortgageInput
    ) -> PCAFDataQuality:
        """Determine PCAF quality for mortgages."""
        if input_data.actual_emissions_co2e is not None and input_data.verified:
            return PCAFDataQuality.SCORE_1
        if input_data.actual_energy_kwh is not None:
            return PCAFDataQuality.SCORE_1
        if input_data.epc_rating is not None:
            return PCAFDataQuality.SCORE_2
        if input_data.floor_area_m2 is not None:
            return PCAFDataQuality.SCORE_3
        return PCAFDataQuality.SCORE_5

    def _determine_pcaf_quality_vehicle(
        self, input_data: MotorVehicleLoanInput
    ) -> PCAFDataQuality:
        """Determine PCAF quality for motor vehicles."""
        if input_data.actual_emissions_co2e is not None and input_data.verified:
            return PCAFDataQuality.SCORE_1
        if input_data.actual_fuel_litres is not None:
            return PCAFDataQuality.SCORE_1
        if input_data.make and input_data.model_name and input_data.year:
            return PCAFDataQuality.SCORE_2
        if input_data.annual_distance_km is not None:
            return PCAFDataQuality.SCORE_3
        return PCAFDataQuality.SCORE_4

    # =========================================================================
    # CRE CALCULATION
    # =========================================================================

    def calculate_cre(
        self, input_data: CREInvestmentInput
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for commercial real estate.

        Data quality paths:
            Score 1: actual_emissions_co2e (verified) or actual_energy_kwh
            Score 2: EPC rating / GRESB score
            Score 3: floor_area_m2 + EUI benchmark
            Score 4-5: Default per-m2 or per-asset average

        Args:
            input_data: CREInvestmentInput.

        Returns:
            InvestmentCalculationResult.
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "CRE calculation #%d: type=%s, outstanding=%s %s",
            calc_number,
            input_data.property_type.value,
            input_data.outstanding_amount,
            input_data.currency.value,
        )

        # Convert to USD
        outstanding_usd = self._convert_to_usd(
            input_data.outstanding_amount, input_data.currency
        )
        property_value_usd = self._convert_to_usd(
            input_data.property_value_at_origination, input_data.currency
        )

        # Attribution
        attribution = self._calculate_attribution_factor(
            outstanding_usd, property_value_usd
        )

        # Determine emissions
        asset_emissions: Decimal
        calc_method: str
        calc_details: Dict[str, Any]

        if input_data.actual_emissions_co2e is not None:
            # Score 1: Direct measured emissions
            asset_emissions = input_data.actual_emissions_co2e
            calc_method = "actual_emissions"
            calc_details = {
                "actual_emissions_tco2e": str(asset_emissions),
                "verified": input_data.verified,
            }
        elif input_data.actual_energy_kwh is not None:
            # Score 1: Actual energy x grid EF
            grid_ef = GRID_EMISSION_FACTORS.get(
                input_data.grid_region.upper(),
                GRID_EMISSION_FACTORS["GLOBAL"],
            )
            emissions_kg = self._quantize(
                input_data.actual_energy_kwh * grid_ef
            )
            asset_emissions = self._quantize(emissions_kg * _KG_TO_TONNES)
            calc_method = "actual_energy"
            calc_details = {
                "actual_energy_kwh": str(input_data.actual_energy_kwh),
                "grid_ef_kgco2e_per_kwh": str(grid_ef),
                "emissions_tco2e": str(asset_emissions),
            }
        elif input_data.floor_area_m2 is not None:
            # Score 2-3: Floor area + EUI benchmark
            asset_emissions, calc_details = self._calculate_building_emissions(
                input_data.floor_area_m2,
                input_data.property_type,
                input_data.climate_zone,
                input_data.grid_region,
            )
            # Apply EPC rating if available
            if input_data.epc_rating is not None:
                epc_mult = EPC_RATING_MULTIPLIERS.get(
                    input_data.epc_rating, _ONE
                )
                asset_emissions = self._quantize(asset_emissions * epc_mult)
                calc_details["epc_rating"] = input_data.epc_rating.value
                calc_details["epc_multiplier"] = str(epc_mult)
                calc_details["adjusted_emissions_tco2e"] = str(asset_emissions)

            calc_method = "floor_area_eui"
        else:
            # Score 4-5: Default per-asset estimate
            intensity = DEFAULT_PROPERTY_INTENSITY["commercial"]
            # Use outstanding as proxy for property size
            outstanding_m_usd = self._quantize(
                outstanding_usd / Decimal("1000000")
            )
            asset_emissions = self._quantize(
                outstanding_m_usd * intensity * _KG_TO_TONNES
            )
            calc_method = "default_intensity"
            calc_details = {
                "default_intensity_kgco2e_per_m2": str(intensity),
                "outstanding_m_usd": str(outstanding_m_usd),
            }

        # Financed emissions
        financed_emissions = self._quantize(attribution * asset_emissions)

        # PCAF quality
        pcaf_score = self._determine_pcaf_quality_cre(input_data)

        # Uncertainty
        lower, upper = self._calculate_uncertainty(financed_emissions, pcaf_score)

        # Duration
        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        # Provenance
        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(asset_emissions),
            str(financed_emissions),
        )

        result = InvestmentCalculationResult(
            asset_type="commercial_real_estate",
            outstanding_amount=input_data.outstanding_amount,
            outstanding_amount_usd=outstanding_usd,
            attribution_factor=attribution,
            denominator_type="property_value_at_origination",
            denominator_value=property_value_usd,
            asset_emissions_co2e=asset_emissions,
            financed_emissions_co2e=financed_emissions,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            calculation_method=calc_method,
            calculation_details=calc_details,
            reporting_year=input_data.reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

        logger.info(
            "CRE #%d complete: attribution=%.6f, asset_co2e=%s, "
            "financed_co2e=%s tCO2e, pcaf=%d, method=%s, duration=%.2fms",
            calc_number, float(attribution), asset_emissions,
            financed_emissions, pcaf_score.value, calc_method,
            float(duration_ms),
        )

        return result

    # =========================================================================
    # MORTGAGE CALCULATION
    # =========================================================================

    def calculate_mortgage(
        self, input_data: MortgageInput
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for a residential mortgage.

        EPC ratings (A-G) map to EUI multipliers:
            A: 0.4x, B: 0.6x, C: 0.8x, D: 1.0x, E: 1.2x, F: 1.5x, G: 2.0x

        Args:
            input_data: MortgageInput.

        Returns:
            InvestmentCalculationResult.
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "Mortgage calculation #%d: type=%s, outstanding=%s %s",
            calc_number,
            input_data.property_type.value,
            input_data.outstanding_loan,
            input_data.currency.value,
        )

        # Convert to USD
        outstanding_usd = self._convert_to_usd(
            input_data.outstanding_loan, input_data.currency
        )
        property_value_usd = self._convert_to_usd(
            input_data.property_value_at_origination, input_data.currency
        )

        # Attribution
        attribution = self._calculate_attribution_factor(
            outstanding_usd, property_value_usd
        )

        # Determine emissions
        asset_emissions: Decimal
        calc_method: str
        calc_details: Dict[str, Any]

        if input_data.actual_emissions_co2e is not None:
            asset_emissions = input_data.actual_emissions_co2e
            calc_method = "actual_emissions"
            calc_details = {
                "actual_emissions_tco2e": str(asset_emissions),
                "verified": input_data.verified,
            }
        elif input_data.actual_energy_kwh is not None:
            grid_ef = GRID_EMISSION_FACTORS.get(
                input_data.grid_region.upper(),
                GRID_EMISSION_FACTORS["GLOBAL"],
            )
            emissions_kg = self._quantize(
                input_data.actual_energy_kwh * grid_ef
            )
            asset_emissions = self._quantize(emissions_kg * _KG_TO_TONNES)
            calc_method = "actual_energy"
            calc_details = {
                "actual_energy_kwh": str(input_data.actual_energy_kwh),
                "grid_ef": str(grid_ef),
                "emissions_tco2e": str(asset_emissions),
            }
        elif input_data.floor_area_m2 is not None:
            # Use residential EUI + EPC adjustment
            asset_emissions, calc_details = self._calculate_building_emissions(
                input_data.floor_area_m2,
                input_data.property_type,
                input_data.climate_zone,
                input_data.grid_region,
            )
            # Apply EPC rating
            if input_data.epc_rating is not None:
                epc_mult = EPC_RATING_MULTIPLIERS.get(
                    input_data.epc_rating, _ONE
                )
                asset_emissions = self._quantize(asset_emissions * epc_mult)
                calc_details["epc_rating"] = input_data.epc_rating.value
                calc_details["epc_multiplier"] = str(epc_mult)
                calc_details["adjusted_emissions_tco2e"] = str(asset_emissions)

            calc_method = "floor_area_eui_epc"
        else:
            # Default per-property
            intensity = DEFAULT_PROPERTY_INTENSITY["residential"]
            outstanding_m_usd = self._quantize(
                outstanding_usd / Decimal("1000000")
            )
            asset_emissions = self._quantize(
                outstanding_m_usd * intensity * _KG_TO_TONNES
            )
            calc_method = "default_intensity"
            calc_details = {
                "default_intensity_kgco2e_per_m2": str(intensity),
            }

        # Financed emissions
        financed_emissions = self._quantize(attribution * asset_emissions)

        # PCAF quality
        pcaf_score = self._determine_pcaf_quality_mortgage(input_data)

        # Uncertainty
        lower, upper = self._calculate_uncertainty(financed_emissions, pcaf_score)

        # Duration
        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        # Provenance
        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(asset_emissions),
            str(financed_emissions),
        )

        result = InvestmentCalculationResult(
            asset_type="mortgage",
            outstanding_amount=input_data.outstanding_loan,
            outstanding_amount_usd=outstanding_usd,
            attribution_factor=attribution,
            denominator_type="property_value_at_origination",
            denominator_value=property_value_usd,
            asset_emissions_co2e=asset_emissions,
            financed_emissions_co2e=financed_emissions,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            calculation_method=calc_method,
            calculation_details=calc_details,
            reporting_year=input_data.reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

        logger.info(
            "Mortgage #%d complete: attribution=%.6f, asset_co2e=%s, "
            "financed_co2e=%s tCO2e, pcaf=%d, method=%s, duration=%.2fms",
            calc_number, float(attribution), asset_emissions,
            financed_emissions, pcaf_score.value, calc_method,
            float(duration_ms),
        )

        return result

    # =========================================================================
    # MOTOR VEHICLE LOAN CALCULATION
    # =========================================================================

    def calculate_motor_vehicle(
        self, input_data: MotorVehicleLoanInput
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for a motor vehicle loan.

        Vehicle categories with annual emissions defaults:
            Passenger car: ~4,600 kgCO2e/yr
            Light commercial: ~6,200 kgCO2e/yr
            Heavy commercial: ~18,500 kgCO2e/yr
            Electric vehicle: ~1,200 kgCO2e/yr (grid-dependent)

        Args:
            input_data: MotorVehicleLoanInput.

        Returns:
            InvestmentCalculationResult.
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "Vehicle loan calculation #%d: category=%s, fuel=%s, "
            "outstanding=%s %s",
            calc_number,
            input_data.vehicle_category.value,
            input_data.fuel_type.value,
            input_data.outstanding_loan,
            input_data.currency.value,
        )

        # Convert to USD
        outstanding_usd = self._convert_to_usd(
            input_data.outstanding_loan, input_data.currency
        )
        vehicle_value_usd = self._convert_to_usd(
            input_data.vehicle_value_at_origination, input_data.currency
        )

        # Attribution
        attribution = self._calculate_attribution_factor(
            outstanding_usd, vehicle_value_usd
        )

        # Determine emissions
        asset_emissions: Decimal
        calc_method: str
        calc_details: Dict[str, Any]

        if input_data.actual_emissions_co2e is not None:
            # Direct measurement (kgCO2e -> tCO2e)
            asset_emissions = self._quantize(
                input_data.actual_emissions_co2e * _KG_TO_TONNES
            )
            calc_method = "actual_emissions"
            calc_details = {
                "actual_emissions_kgco2e": str(input_data.actual_emissions_co2e),
                "actual_emissions_tco2e": str(asset_emissions),
                "verified": input_data.verified,
            }
        else:
            # Distance-based or category average
            asset_emissions, calc_details = self._calculate_vehicle_emissions(
                input_data.vehicle_category,
                input_data.annual_distance_km,
                input_data.fuel_type,
            )
            calc_method = calc_details.get("method", "category_average")

        # Financed emissions
        financed_emissions = self._quantize(attribution * asset_emissions)

        # PCAF quality
        pcaf_score = self._determine_pcaf_quality_vehicle(input_data)

        # Uncertainty
        lower, upper = self._calculate_uncertainty(financed_emissions, pcaf_score)

        # Duration
        duration_ms = Decimal(str((time.monotonic() - start_time) * 1000))

        # Provenance
        provenance = _compute_hash(
            input_data.model_dump(mode="json"),
            str(attribution),
            str(asset_emissions),
            str(financed_emissions),
        )

        result = InvestmentCalculationResult(
            asset_type="motor_vehicle_loan",
            outstanding_amount=input_data.outstanding_loan,
            outstanding_amount_usd=outstanding_usd,
            attribution_factor=attribution,
            denominator_type="vehicle_value_at_origination",
            denominator_value=vehicle_value_usd,
            asset_emissions_co2e=asset_emissions,
            financed_emissions_co2e=financed_emissions,
            pcaf_quality_score=pcaf_score.value,
            uncertainty_lower_co2e=lower,
            uncertainty_upper_co2e=upper,
            calculation_method=calc_method,
            calculation_details=calc_details,
            reporting_year=input_data.reporting_year,
            processing_time_ms=self._quantize_2dp(duration_ms),
            provenance_hash=provenance,
        )

        logger.info(
            "Vehicle loan #%d complete: category=%s, attribution=%.6f, "
            "asset_co2e=%s, financed_co2e=%s tCO2e, pcaf=%d, "
            "method=%s, duration=%.2fms",
            calc_number,
            input_data.vehicle_category.value,
            float(attribution),
            asset_emissions,
            financed_emissions,
            pcaf_score.value,
            calc_method,
            float(duration_ms),
        )

        return result

    # =========================================================================
    # BATCH CALCULATION
    # =========================================================================

    def calculate_batch(
        self,
        inputs: List[Union[CREInvestmentInput, MortgageInput, MotorVehicleLoanInput]],
    ) -> List[InvestmentCalculationResult]:
        """
        Calculate financed emissions for a batch of real assets.

        Routes each input to the appropriate calculator based on type.

        Args:
            inputs: List of CRE, Mortgage, or Vehicle inputs.

        Returns:
            List of successful InvestmentCalculationResult.
        """
        start_time = time.monotonic()
        results: List[InvestmentCalculationResult] = []
        error_count = 0

        logger.info("Batch real asset calculation: %d inputs", len(inputs))

        for i, inp in enumerate(inputs):
            try:
                if isinstance(inp, CREInvestmentInput):
                    result = self.calculate_cre(inp)
                elif isinstance(inp, MortgageInput):
                    result = self.calculate_mortgage(inp)
                elif isinstance(inp, MotorVehicleLoanInput):
                    result = self.calculate_motor_vehicle(inp)
                else:
                    raise TypeError(
                        f"Unsupported input type: {type(inp).__name__}"
                    )
                results.append(result)
            except Exception as exc:
                error_count += 1
                logger.error(
                    "Batch real asset #%d failed: %s",
                    i + 1, exc, exc_info=True,
                )

        total_duration = time.monotonic() - start_time
        logger.info(
            "Batch real asset complete: %d/%d succeeded, "
            "%d errors, duration=%.3fs",
            len(results), len(inputs), error_count, total_duration,
        )

        return results

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def get_calculation_count(self) -> int:
        """Get total calculation count."""
        with self._count_lock:
            return self._calculation_count

    def get_engine_summary(self) -> Dict[str, Any]:
        """Get engine state summary."""
        return {
            "engine": "RealAssetCalculatorEngine",
            "agent_id": AGENT_ID,
            "version": VERSION,
            "calculation_count": self.get_calculation_count(),
            "supported_asset_types": [t.value for t in RealAssetType],
            "property_types": len(PropertyType),
            "climate_zones": len(ClimateZone),
            "grid_regions": len(GRID_EMISSION_FACTORS),
            "vehicle_categories": len(VehicleCategory),
            "epc_ratings": len(EPCRating),
        }

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (testing only)."""
        with cls._lock:
            cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================

_calculator_instance: Optional[RealAssetCalculatorEngine] = None
_calculator_lock: threading.RLock = threading.RLock()


def get_real_asset_calculator() -> RealAssetCalculatorEngine:
    """Get singleton RealAssetCalculatorEngine."""
    global _calculator_instance
    with _calculator_lock:
        if _calculator_instance is None:
            _calculator_instance = RealAssetCalculatorEngine()
        return _calculator_instance


def reset_real_asset_calculator() -> None:
    """Reset module-level calculator (testing only)."""
    global _calculator_instance
    with _calculator_lock:
        _calculator_instance = None
    RealAssetCalculatorEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Enums
    "RealAssetType",
    "PropertyType",
    "ClimateZone",
    "EPCRating",
    "VehicleCategory",
    "VehicleFuelType",
    "PCAFDataQuality",
    "CurrencyCode",
    # Constants
    "CRE_EUI_BENCHMARKS",
    "RESIDENTIAL_EUI_BENCHMARKS",
    "EPC_RATING_MULTIPLIERS",
    "GRID_EMISSION_FACTORS",
    "VEHICLE_ANNUAL_EMISSIONS",
    "VEHICLE_EF_PER_KM",
    "AVERAGE_ANNUAL_DISTANCE_KM",
    "PCAF_UNCERTAINTY_RANGES",
    "CURRENCY_RATES_TO_USD",
    "DEFAULT_PROPERTY_INTENSITY",
    # Input models
    "CREInvestmentInput",
    "MortgageInput",
    "MotorVehicleLoanInput",
    # Result model
    "InvestmentCalculationResult",
    # Engine
    "RealAssetCalculatorEngine",
    "get_real_asset_calculator",
    "reset_real_asset_calculator",
]
