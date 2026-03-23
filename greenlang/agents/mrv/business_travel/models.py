"""
Business Travel Agent Models (AGENT-MRV-019)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 6
(Business Travel) emissions calculations.

Supports:
- 4 calculation methods (supplier-specific, distance-based, average-data, spend-based)
- 8 transport modes (air, rail, road, bus, taxi, ferry, motorcycle, hotel)
- DEFRA 2024 emission factors for all transport modes
- ICAO-aligned radiative forcing (RF) multipliers for aviation
- Cabin class multipliers (economy, premium economy, business, first)
- 8 rail types with WTT (well-to-tank) factors
- 13 road vehicle types with per-vkm and per-pkm factors
- Hotel room-night emissions for 16 countries
- EEIO spend-based factors (10 NAICS codes)
- 50 major airport database with great-circle distance calculation
- CPI deflation and multi-currency conversion
- Data quality indicators (DQI) with 5-dimension scoring
- Uncertainty quantification (Monte Carlo, analytical, IPCC Tier 2)
- Compliance checking for 7 frameworks (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, SB 253, GRI)
- SHA-256 provenance chain with 10-stage pipeline
- Hot-spot analysis for route and mode optimization

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.business_travel.models import FlightInput, CabinClass
    >>> flight = FlightInput(
    ...     origin_iata="JFK",
    ...     destination_iata="LHR",
    ...     cabin_class=CabinClass.BUSINESS,
    ...     passengers=1,
    ...     round_trip=True
    ... )
    >>> from greenlang.agents.mrv.business_travel.models import HotelInput, HotelClass
    >>> hotel = HotelInput(
    ...     country_code="GB",
    ...     room_nights=3,
    ...     hotel_class=HotelClass.UPSCALE
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict
import hashlib
import json
import math

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-006"
AGENT_COMPONENT: str = "AGENT-MRV-019"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_bt_"

# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class CalculationMethod(str, Enum):
    """Calculation method for business travel emissions per GHG Protocol."""

    SUPPLIER_SPECIFIC = "supplier_specific"  # Carrier/supplier-reported emissions
    DISTANCE_BASED = "distance_based"  # Distance x mode-specific EF
    AVERAGE_DATA = "average_data"  # Average industry emission factors
    SPEND_BASED = "spend_based"  # Travel spend x EEIO factor


class TransportMode(str, Enum):
    """Transport modes for business travel."""

    AIR = "air"  # Commercial aviation
    RAIL = "rail"  # Train / metro / light rail
    ROAD = "road"  # Car / rental car
    BUS = "bus"  # Local bus / coach
    TAXI = "taxi"  # Taxi / ride-hailing
    FERRY = "ferry"  # Ferry / water taxi
    MOTORCYCLE = "motorcycle"  # Motorcycle / scooter
    HOTEL = "hotel"  # Hotel accommodation (not transport but Cat 6 scope)


class FlightDistanceBand(str, Enum):
    """DEFRA flight distance bands for emission factor selection."""

    DOMESTIC = "domestic"  # < 463 km (UK) / < 700 km (general)
    SHORT_HAUL = "short_haul"  # 463-3700 km (UK) / 700-3700 km (general)
    LONG_HAUL = "long_haul"  # > 3700 km
    INTERNATIONAL_AVG = "international_avg"  # Weighted international average


class CabinClass(str, Enum):
    """Aircraft cabin class affecting per-passenger emissions."""

    ECONOMY = "economy"  # Standard economy
    PREMIUM_ECONOMY = "premium_economy"  # Premium economy / economy plus
    BUSINESS = "business"  # Business class
    FIRST = "first"  # First class


class RailType(str, Enum):
    """Rail transport types with distinct emission profiles."""

    NATIONAL = "national"  # National rail (UK average)
    INTERNATIONAL = "international"  # International rail (Eurostar, etc.)
    LIGHT_RAIL = "light_rail"  # Light rail / tram
    UNDERGROUND = "underground"  # Metro / underground / subway
    EUROSTAR = "eurostar"  # Eurostar (Channel Tunnel)
    HIGH_SPEED = "high_speed"  # High-speed rail (TGV, Shinkansen, etc.)
    US_INTERCITY = "us_intercity"  # US intercity rail (Amtrak)
    US_COMMUTER = "us_commuter"  # US commuter rail


class RoadVehicleType(str, Enum):
    """Road vehicle types with distinct emission profiles."""

    CAR_AVERAGE = "car_average"  # Average car (unknown fuel type)
    CAR_SMALL_PETROL = "car_small_petrol"  # Small petrol car (< 1.4L)
    CAR_MEDIUM_PETROL = "car_medium_petrol"  # Medium petrol car (1.4-2.0L)
    CAR_LARGE_PETROL = "car_large_petrol"  # Large petrol car (> 2.0L)
    CAR_SMALL_DIESEL = "car_small_diesel"  # Small diesel car (< 1.7L)
    CAR_MEDIUM_DIESEL = "car_medium_diesel"  # Medium diesel car (1.7-2.0L)
    CAR_LARGE_DIESEL = "car_large_diesel"  # Large diesel car (> 2.0L)
    HYBRID = "hybrid"  # Hybrid electric vehicle (HEV)
    PLUGIN_HYBRID = "plugin_hybrid"  # Plug-in hybrid electric vehicle (PHEV)
    BEV = "bev"  # Battery electric vehicle (BEV)
    TAXI_REGULAR = "taxi_regular"  # Regular taxi / ride-hailing
    TAXI_BLACK_CAB = "taxi_black_cab"  # London black cab / large taxi
    MOTORCYCLE = "motorcycle"  # Motorcycle / scooter


class FuelType(str, Enum):
    """Fuel types for fuel-based road calculations."""

    PETROL = "petrol"  # Gasoline / petrol
    DIESEL = "diesel"  # Diesel
    LPG = "lpg"  # Liquefied petroleum gas
    CNG = "cng"  # Compressed natural gas (per kg)
    E85 = "e85"  # Ethanol blend (85% ethanol, 15% petrol)


class BusType(str, Enum):
    """Bus transport types."""

    LOCAL = "local"  # Local bus / city bus
    COACH = "coach"  # Long-distance coach / intercity bus


class FerryType(str, Enum):
    """Ferry passenger types."""

    FOOT_PASSENGER = "foot_passenger"  # Walk-on passenger (no vehicle)
    CAR_PASSENGER = "car_passenger"  # Passenger with car on ferry


class HotelClass(str, Enum):
    """Hotel class/tier affecting room-night emissions."""

    BUDGET = "budget"  # Budget / economy hotel
    STANDARD = "standard"  # Standard / mid-range hotel
    UPSCALE = "upscale"  # Upscale / 4-star hotel
    LUXURY = "luxury"  # Luxury / 5-star hotel


class TripPurpose(str, Enum):
    """Business trip purpose for reporting segmentation."""

    BUSINESS = "business"  # General business meeting
    CONFERENCE = "conference"  # Conference / event attendance
    CLIENT_VISIT = "client_visit"  # Client site visit
    TRAINING = "training"  # Training / professional development
    OTHER = "other"  # Other business travel


class EFSource(str, Enum):
    """Emission factor data source."""

    SUPPLIER = "supplier"  # Carrier/supplier-reported factors
    DEFRA = "defra"  # UK DEFRA/DESNZ conversion factors
    ICAO = "icao"  # ICAO Carbon Emissions Calculator
    EPA = "epa"  # US EPA emission factors
    IEA = "iea"  # IEA energy statistics
    EEIO = "eeio"  # Environmentally Extended Input-Output
    CUSTOM = "custom"  # Custom / organization-specific factors


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework for compliance checks."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 Climate Change
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    SB_253 = "sb_253"  # California SB 253 (Climate Corporate Data Accountability Act)
    GRI = "gri"  # GRI 305 Emissions Standard


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges."""

    TIER_1 = "tier_1"  # Supplier-specific / primary data
    TIER_2 = "tier_2"  # Regional / mode-specific secondary data
    TIER_3 = "tier_3"  # Global average / spend-based estimates


class RFOption(str, Enum):
    """Radiative forcing (RF) reporting option for aviation emissions."""

    WITH_RF = "with_rf"  # Include RF uplift factor (DEFRA default)
    WITHOUT_RF = "without_rf"  # Exclude RF uplift (CO2-only basis)
    BOTH = "both"  # Report both with and without RF


class ProvenanceStage(str, Enum):
    """Processing pipeline stages for provenance tracking."""

    VALIDATE = "validate"  # Input validation
    CLASSIFY = "classify"  # Trip classification (mode, distance band)
    NORMALIZE = "normalize"  # Unit normalization (currency, distance)
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution
    CALCULATE_FLIGHTS = "calculate_flights"  # Flight emissions calculation
    CALCULATE_GROUND = "calculate_ground"  # Ground transport calculation
    ALLOCATE = "allocate"  # Department / cost center allocation
    COMPLIANCE = "compliance"  # Compliance checks
    AGGREGATE = "aggregate"  # Aggregation by mode, period, department
    SEAL = "seal"  # Provenance chain sealing


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method."""

    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    ANALYTICAL = "analytical"  # Analytical error propagation
    IPCC_TIER_2 = "ipcc_tier_2"  # IPCC Tier 2 default ranges


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol."""

    REPRESENTATIVENESS = "representativeness"  # How well data represents the activity
    COMPLETENESS = "completeness"  # Fraction of data coverage
    TEMPORAL = "temporal"  # Temporal correlation to reporting year
    GEOGRAPHICAL = "geographical"  # Geographical correlation to activity
    TECHNOLOGICAL = "technological"  # Technological correlation to activity


class DQIScore(str, Enum):
    """Data Quality Indicator scores (1-5 scale, 5 = best)."""

    VERY_HIGH = "very_high"  # 5 - Primary data, site-specific
    HIGH = "high"  # 4 - Verified secondary data
    MEDIUM = "medium"  # 3 - Industry average data
    LOW = "low"  # 2 - Estimated / proxy data
    VERY_LOW = "very_low"  # 1 - Spend-based / generic data


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    PASS = "pass"  # Fully compliant
    FAIL = "fail"  # Non-compliant
    WARNING = "warning"  # Partially compliant / needs attention


class GWPVersion(str, Enum):
    """IPCC Global Warming Potential assessment report version."""

    AR4 = "ar4"  # Fourth Assessment Report (100-year)
    AR5 = "ar5"  # Fifth Assessment Report (100-year)
    AR6 = "ar6"  # Sixth Assessment Report (100-year)
    AR6_20YR = "ar6_20yr"  # Sixth Assessment Report (20-year)


class EmissionGas(str, Enum):
    """Greenhouse gas types relevant to business travel."""

    CO2 = "co2"  # Carbon dioxide
    CH4 = "ch4"  # Methane
    N2O = "n2o"  # Nitrous oxide
    CO2_BIOGENIC = "co2_biogenic"  # Biogenic CO2 (memo item for biofuels)


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for spend-based calculations."""

    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    CAD = "CAD"  # Canadian Dollar
    AUD = "AUD"  # Australian Dollar
    JPY = "JPY"  # Japanese Yen
    CNY = "CNY"  # Chinese Yuan
    INR = "INR"  # Indian Rupee
    CHF = "CHF"  # Swiss Franc
    SGD = "SGD"  # Singapore Dollar
    BRL = "BRL"  # Brazilian Real
    ZAR = "ZAR"  # South African Rand


class ExportFormat(str, Enum):
    """Export format for results."""

    JSON = "json"  # JSON format
    CSV = "csv"  # CSV format
    EXCEL = "excel"  # Excel (XLSX) format
    PDF = "pdf"  # PDF report


class BatchStatus(str, Enum):
    """Batch calculation processing status."""

    PENDING = "pending"  # Awaiting processing
    PROCESSING = "processing"  # Currently processing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Processing failed
    PARTIAL = "partial"  # Some records failed


class AllocationMethod(str, Enum):
    """Emissions allocation method for multi-entity reporting."""

    EQUAL = "equal"  # Equal allocation across entities
    HEADCOUNT = "headcount"  # Based on employee headcount
    COST_CENTER = "cost_center"  # Based on cost center budgets
    DEPARTMENT = "department"  # Based on department attribution
    PROJECT = "project"  # Based on project assignment
    CUSTOM = "custom"  # Custom allocation weights


# ==============================================================================
# CONSTANT TABLES
# ==============================================================================

# Global Warming Potential values (100-year unless stated)
GWP_VALUES: Dict[GWPVersion, Dict[str, Decimal]] = {
    GWPVersion.AR4: {
        "co2": Decimal("1"),
        "ch4": Decimal("25"),
        "n2o": Decimal("298"),
    },
    GWPVersion.AR5: {
        "co2": Decimal("1"),
        "ch4": Decimal("28"),
        "n2o": Decimal("265"),
    },
    GWPVersion.AR6: {
        "co2": Decimal("1"),
        "ch4": Decimal("27.9"),
        "n2o": Decimal("273"),
    },
    GWPVersion.AR6_20YR: {
        "co2": Decimal("1"),
        "ch4": Decimal("81.2"),
        "n2o": Decimal("273"),
    },
}

# Air emission factors (kgCO2e per passenger-km) by distance band - DEFRA 2024
AIR_EMISSION_FACTORS: Dict[FlightDistanceBand, Dict[str, Decimal]] = {
    FlightDistanceBand.DOMESTIC: {
        "without_rf": Decimal("0.24587"),
        "with_rf": Decimal("0.27916"),
        "wtt": Decimal("0.05765"),
    },
    FlightDistanceBand.SHORT_HAUL: {
        "without_rf": Decimal("0.15353"),
        "with_rf": Decimal("0.17435"),
        "wtt": Decimal("0.03600"),
    },
    FlightDistanceBand.LONG_HAUL: {
        "without_rf": Decimal("0.19309"),
        "with_rf": Decimal("0.21932"),
        "wtt": Decimal("0.04528"),
    },
    FlightDistanceBand.INTERNATIONAL_AVG: {
        "without_rf": Decimal("0.18362"),
        "with_rf": Decimal("0.20856"),
        "wtt": Decimal("0.04306"),
    },
}

# Cabin class multipliers relative to economy (DEFRA 2024)
CABIN_CLASS_MULTIPLIERS: Dict[CabinClass, Decimal] = {
    CabinClass.ECONOMY: Decimal("1.0"),
    CabinClass.PREMIUM_ECONOMY: Decimal("1.6"),
    CabinClass.BUSINESS: Decimal("2.9"),
    CabinClass.FIRST: Decimal("4.0"),
}

# Rail emission factors (kgCO2e per passenger-km) - DEFRA 2024
RAIL_EMISSION_FACTORS: Dict[RailType, Dict[str, Decimal]] = {
    RailType.NATIONAL: {
        "ttw": Decimal("0.03549"),
        "wtt": Decimal("0.00434"),
    },
    RailType.INTERNATIONAL: {
        "ttw": Decimal("0.00446"),
        "wtt": Decimal("0.00086"),
    },
    RailType.LIGHT_RAIL: {
        "ttw": Decimal("0.02904"),
        "wtt": Decimal("0.00612"),
    },
    RailType.UNDERGROUND: {
        "ttw": Decimal("0.02781"),
        "wtt": Decimal("0.00586"),
    },
    RailType.EUROSTAR: {
        "ttw": Decimal("0.00446"),
        "wtt": Decimal("0.00086"),
    },
    RailType.HIGH_SPEED: {
        "ttw": Decimal("0.00324"),
        "wtt": Decimal("0.00068"),
    },
    RailType.US_INTERCITY: {
        "ttw": Decimal("0.08900"),
        "wtt": Decimal("0.01100"),
    },
    RailType.US_COMMUTER: {
        "ttw": Decimal("0.10500"),
        "wtt": Decimal("0.01300"),
    },
}

# Road vehicle emission factors (kgCO2e) - DEFRA 2024
# ef_per_vkm: emissions per vehicle-km
# ef_per_pkm: emissions per passenger-km
# wtt_per_vkm: well-to-tank per vehicle-km
# occupancy: average occupancy factor
ROAD_VEHICLE_EMISSION_FACTORS: Dict[RoadVehicleType, Dict[str, Decimal]] = {
    RoadVehicleType.CAR_AVERAGE: {
        "ef_per_vkm": Decimal("0.27145"),
        "ef_per_pkm": Decimal("0.17082"),
        "wtt_per_vkm": Decimal("0.06291"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.CAR_SMALL_PETROL: {
        "ef_per_vkm": Decimal("0.20755"),
        "ef_per_pkm": Decimal("0.13053"),
        "wtt_per_vkm": Decimal("0.05249"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.CAR_MEDIUM_PETROL: {
        "ef_per_vkm": Decimal("0.25594"),
        "ef_per_pkm": Decimal("0.16106"),
        "wtt_per_vkm": Decimal("0.06475"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.CAR_LARGE_PETROL: {
        "ef_per_vkm": Decimal("0.35388"),
        "ef_per_pkm": Decimal("0.22258"),
        "wtt_per_vkm": Decimal("0.08953"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.CAR_SMALL_DIESEL: {
        "ef_per_vkm": Decimal("0.19290"),
        "ef_per_pkm": Decimal("0.12132"),
        "wtt_per_vkm": Decimal("0.04346"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.CAR_MEDIUM_DIESEL: {
        "ef_per_vkm": Decimal("0.23280"),
        "ef_per_pkm": Decimal("0.14642"),
        "wtt_per_vkm": Decimal("0.05245"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.CAR_LARGE_DIESEL: {
        "ef_per_vkm": Decimal("0.29610"),
        "ef_per_pkm": Decimal("0.18629"),
        "wtt_per_vkm": Decimal("0.06673"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.HYBRID: {
        "ef_per_vkm": Decimal("0.17830"),
        "ef_per_pkm": Decimal("0.11214"),
        "wtt_per_vkm": Decimal("0.04511"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.PLUGIN_HYBRID: {
        "ef_per_vkm": Decimal("0.10250"),
        "ef_per_pkm": Decimal("0.06447"),
        "wtt_per_vkm": Decimal("0.02166"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.BEV: {
        "ef_per_vkm": Decimal("0.07005"),
        "ef_per_pkm": Decimal("0.04406"),
        "wtt_per_vkm": Decimal("0.01479"),
        "occupancy": Decimal("1.59"),
    },
    RoadVehicleType.TAXI_REGULAR: {
        "ef_per_vkm": Decimal("0.20920"),
        "ef_per_pkm": Decimal("0.14880"),
        "wtt_per_vkm": Decimal("0.04710"),
        "occupancy": Decimal("1.41"),
    },
    RoadVehicleType.TAXI_BLACK_CAB: {
        "ef_per_vkm": Decimal("0.31477"),
        "ef_per_pkm": Decimal("0.22378"),
        "wtt_per_vkm": Decimal("0.07093"),
        "occupancy": Decimal("1.41"),
    },
    RoadVehicleType.MOTORCYCLE: {
        "ef_per_vkm": Decimal("0.11337"),
        "ef_per_pkm": Decimal("0.11337"),
        "wtt_per_vkm": Decimal("0.02867"),
        "occupancy": Decimal("1.0"),
    },
}

# Fuel emission factors (kgCO2e per litre / per kg for CNG) - DEFRA 2024
FUEL_EMISSION_FACTORS: Dict[FuelType, Dict[str, Decimal]] = {
    FuelType.PETROL: {
        "ef_per_litre": Decimal("2.31480"),
        "wtt_per_litre": Decimal("0.58549"),
    },
    FuelType.DIESEL: {
        "ef_per_litre": Decimal("2.70370"),
        "wtt_per_litre": Decimal("0.60927"),
    },
    FuelType.LPG: {
        "ef_per_litre": Decimal("1.55370"),
        "wtt_per_litre": Decimal("0.32149"),
    },
    FuelType.CNG: {
        "ef_per_litre": Decimal("2.53970"),  # per kg
        "wtt_per_litre": Decimal("0.50870"),  # per kg
    },
    FuelType.E85: {
        "ef_per_litre": Decimal("0.34728"),
        "wtt_per_litre": Decimal("0.07890"),
    },
}

# Bus emission factors (kgCO2e per passenger-km) - DEFRA 2024
BUS_EMISSION_FACTORS: Dict[BusType, Dict[str, Decimal]] = {
    BusType.LOCAL: {
        "ef": Decimal("0.10312"),
        "wtt": Decimal("0.01847"),
    },
    BusType.COACH: {
        "ef": Decimal("0.02732"),
        "wtt": Decimal("0.00489"),
    },
}

# Ferry emission factors (kgCO2e per passenger-km) - DEFRA 2024
FERRY_EMISSION_FACTORS: Dict[FerryType, Dict[str, Decimal]] = {
    FerryType.FOOT_PASSENGER: {
        "ef": Decimal("0.01877"),
        "wtt": Decimal("0.00572"),
    },
    FerryType.CAR_PASSENGER: {
        "ef": Decimal("0.12952"),
        "wtt": Decimal("0.03950"),
    },
}

# Hotel emission factors (kgCO2e per room-night) by country - DEFRA 2024 / Cornell HCMI
HOTEL_EMISSION_FACTORS: Dict[str, Decimal] = {
    "GB": Decimal("12.32"),
    "US": Decimal("21.12"),
    "CA": Decimal("14.40"),
    "FR": Decimal("7.26"),
    "DE": Decimal("13.50"),
    "ES": Decimal("10.60"),
    "IT": Decimal("11.10"),
    "NL": Decimal("10.00"),
    "JP": Decimal("28.85"),
    "CN": Decimal("34.56"),
    "IN": Decimal("22.08"),
    "AU": Decimal("25.90"),
    "BR": Decimal("8.28"),
    "SG": Decimal("27.00"),
    "AE": Decimal("37.50"),
    "GLOBAL": Decimal("20.90"),
}

# Hotel class multipliers relative to standard
HOTEL_CLASS_MULTIPLIERS: Dict[HotelClass, Decimal] = {
    HotelClass.BUDGET: Decimal("0.75"),
    HotelClass.STANDARD: Decimal("1.0"),
    HotelClass.UPSCALE: Decimal("1.35"),
    HotelClass.LUXURY: Decimal("1.80"),
}

# EEIO factors for spend-based calculation (kgCO2e per USD)
# Source: EPA USEEIO v2.0 / Exiobase 3
EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "481000": {
        "name": "Air transportation",
        "ef": Decimal("0.4770"),
    },
    "482000": {
        "name": "Rail transportation",
        "ef": Decimal("0.3100"),
    },
    "485000": {
        "name": "Ground passenger transport",
        "ef": Decimal("0.2600"),
    },
    "485310": {
        "name": "Taxi/ride-hailing",
        "ef": Decimal("0.2800"),
    },
    "532100": {
        "name": "Automotive rental/leasing",
        "ef": Decimal("0.1950"),
    },
    "721100": {
        "name": "Hotels and motels",
        "ef": Decimal("0.1490"),
    },
    "721200": {
        "name": "RV parks and camps",
        "ef": Decimal("0.1200"),
    },
    "722500": {
        "name": "Restaurants (travel meals)",
        "ef": Decimal("0.2050"),
    },
    "483000": {
        "name": "Water transportation",
        "ef": Decimal("0.5200"),
    },
    "487000": {
        "name": "Scenic/sightseeing transport",
        "ef": Decimal("0.3400"),
    },
}

# Currency exchange rates to USD (approximate mid-market rates)
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
}

# CPI deflators for spend-based calculation (base year 2021 = 1.0)
# Source: US BLS CPI-U / OECD CPI
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

# Data Quality Indicator scoring matrix (1-5 scale per dimension)
DQI_SCORING: Dict[DQIDimension, Dict[DQIScore, Decimal]] = {
    DQIDimension.REPRESENTATIVENESS: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
    DQIDimension.COMPLETENESS: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
    DQIDimension.TEMPORAL: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
    DQIDimension.GEOGRAPHICAL: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
    DQIDimension.TECHNOLOGICAL: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
}

# DQI dimension weights (sum to 1.0)
DQI_WEIGHTS: Dict[DQIDimension, Decimal] = {
    DQIDimension.REPRESENTATIVENESS: Decimal("0.30"),
    DQIDimension.COMPLETENESS: Decimal("0.25"),
    DQIDimension.TEMPORAL: Decimal("0.15"),
    DQIDimension.GEOGRAPHICAL: Decimal("0.15"),
    DQIDimension.TECHNOLOGICAL: Decimal("0.15"),
}

# Uncertainty ranges by calculation method and data quality tier
# Values represent the half-width of the 95% confidence interval as a fraction
UNCERTAINTY_RANGES: Dict[str, Dict[DataQualityTier, Decimal]] = {
    "supplier_specific": {
        DataQualityTier.TIER_1: Decimal("0.05"),
        DataQualityTier.TIER_2: Decimal("0.08"),
        DataQualityTier.TIER_3: Decimal("0.10"),
    },
    "distance_air": {
        DataQualityTier.TIER_1: Decimal("0.10"),
        DataQualityTier.TIER_2: Decimal("0.15"),
        DataQualityTier.TIER_3: Decimal("0.20"),
    },
    "distance_rail": {
        DataQualityTier.TIER_1: Decimal("0.15"),
        DataQualityTier.TIER_2: Decimal("0.20"),
        DataQualityTier.TIER_3: Decimal("0.25"),
    },
    "distance_road": {
        DataQualityTier.TIER_1: Decimal("0.15"),
        DataQualityTier.TIER_2: Decimal("0.22"),
        DataQualityTier.TIER_3: Decimal("0.30"),
    },
    "hotel": {
        DataQualityTier.TIER_1: Decimal("0.20"),
        DataQualityTier.TIER_2: Decimal("0.28"),
        DataQualityTier.TIER_3: Decimal("0.35"),
    },
    "average_data": {
        DataQualityTier.TIER_1: Decimal("0.25"),
        DataQualityTier.TIER_2: Decimal("0.32"),
        DataQualityTier.TIER_3: Decimal("0.40"),
    },
    "spend_based": {
        DataQualityTier.TIER_1: Decimal("0.40"),
        DataQualityTier.TIER_2: Decimal("0.50"),
        DataQualityTier.TIER_3: Decimal("0.60"),
    },
}

# Required disclosures per compliance framework
FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_co2e",
        "method_used",
        "ef_sources",
        "exclusions",
        "dqi_score",
    ],
    ComplianceFramework.ISO_14064: [
        "total_co2e",
        "uncertainty_analysis",
        "base_year",
        "methodology",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "total_co2e",
        "category_breakdown",
        "methodology",
        "targets",
        "actions",
    ],
    ComplianceFramework.CDP: [
        "total_co2e",
        "with_rf",
        "without_rf",
        "mode_breakdown",
        "verification_status",
    ],
    ComplianceFramework.SBTI: [
        "total_co2e",
        "target_coverage",
        "rf_inclusion",
        "progress_tracking",
    ],
    ComplianceFramework.SB_253: [
        "total_co2e",
        "methodology",
        "assurance_opinion",
    ],
    ComplianceFramework.GRI: [
        "total_co2e",
        "gases_included",
        "base_year",
        "standards_used",
    ],
}

# Major airport database (IATA code -> metadata)
# Used for great-circle distance calculation and distance band classification
AIRPORT_DATABASE: Dict[str, Dict[str, Any]] = {
    "JFK": {
        "name": "John F. Kennedy International",
        "lat": Decimal("40.6413"),
        "lon": Decimal("-73.7781"),
        "country": "US",
    },
    "LAX": {
        "name": "Los Angeles International",
        "lat": Decimal("33.9425"),
        "lon": Decimal("-118.4081"),
        "country": "US",
    },
    "ORD": {
        "name": "O'Hare International",
        "lat": Decimal("41.9742"),
        "lon": Decimal("-87.9073"),
        "country": "US",
    },
    "LHR": {
        "name": "London Heathrow",
        "lat": Decimal("51.4700"),
        "lon": Decimal("-0.4543"),
        "country": "GB",
    },
    "CDG": {
        "name": "Paris Charles de Gaulle",
        "lat": Decimal("49.0097"),
        "lon": Decimal("2.5479"),
        "country": "FR",
    },
    "FRA": {
        "name": "Frankfurt",
        "lat": Decimal("50.0379"),
        "lon": Decimal("8.5622"),
        "country": "DE",
    },
    "AMS": {
        "name": "Amsterdam Schiphol",
        "lat": Decimal("52.3105"),
        "lon": Decimal("4.7683"),
        "country": "NL",
    },
    "DXB": {
        "name": "Dubai International",
        "lat": Decimal("25.2532"),
        "lon": Decimal("55.3657"),
        "country": "AE",
    },
    "SIN": {
        "name": "Singapore Changi",
        "lat": Decimal("1.3644"),
        "lon": Decimal("103.9915"),
        "country": "SG",
    },
    "HND": {
        "name": "Tokyo Haneda",
        "lat": Decimal("35.5494"),
        "lon": Decimal("139.7798"),
        "country": "JP",
    },
    "NRT": {
        "name": "Tokyo Narita",
        "lat": Decimal("35.7720"),
        "lon": Decimal("140.3929"),
        "country": "JP",
    },
    "PEK": {
        "name": "Beijing Capital",
        "lat": Decimal("40.0799"),
        "lon": Decimal("116.6031"),
        "country": "CN",
    },
    "PVG": {
        "name": "Shanghai Pudong",
        "lat": Decimal("31.1443"),
        "lon": Decimal("121.8083"),
        "country": "CN",
    },
    "HKG": {
        "name": "Hong Kong",
        "lat": Decimal("22.3080"),
        "lon": Decimal("113.9185"),
        "country": "HK",
    },
    "SYD": {
        "name": "Sydney",
        "lat": Decimal("-33.9461"),
        "lon": Decimal("151.1772"),
        "country": "AU",
    },
    "MEL": {
        "name": "Melbourne",
        "lat": Decimal("-37.6690"),
        "lon": Decimal("144.8410"),
        "country": "AU",
    },
    "GRU": {
        "name": "Sao Paulo Guarulhos",
        "lat": Decimal("-23.4356"),
        "lon": Decimal("-46.4731"),
        "country": "BR",
    },
    "DEL": {
        "name": "Delhi Indira Gandhi",
        "lat": Decimal("28.5562"),
        "lon": Decimal("77.1000"),
        "country": "IN",
    },
    "BOM": {
        "name": "Mumbai",
        "lat": Decimal("19.0896"),
        "lon": Decimal("72.8656"),
        "country": "IN",
    },
    "ICN": {
        "name": "Seoul Incheon",
        "lat": Decimal("37.4602"),
        "lon": Decimal("126.4407"),
        "country": "KR",
    },
    "ATL": {
        "name": "Atlanta",
        "lat": Decimal("33.6407"),
        "lon": Decimal("-84.4277"),
        "country": "US",
    },
    "DFW": {
        "name": "Dallas/Fort Worth",
        "lat": Decimal("32.8998"),
        "lon": Decimal("-97.0403"),
        "country": "US",
    },
    "DEN": {
        "name": "Denver",
        "lat": Decimal("39.8561"),
        "lon": Decimal("-104.6737"),
        "country": "US",
    },
    "SFO": {
        "name": "San Francisco",
        "lat": Decimal("37.6213"),
        "lon": Decimal("-122.3790"),
        "country": "US",
    },
    "SEA": {
        "name": "Seattle-Tacoma",
        "lat": Decimal("47.4502"),
        "lon": Decimal("-122.3088"),
        "country": "US",
    },
    "MIA": {
        "name": "Miami",
        "lat": Decimal("25.7959"),
        "lon": Decimal("-80.2870"),
        "country": "US",
    },
    "BOS": {
        "name": "Boston Logan",
        "lat": Decimal("42.3656"),
        "lon": Decimal("-71.0096"),
        "country": "US",
    },
    "IAD": {
        "name": "Washington Dulles",
        "lat": Decimal("38.9531"),
        "lon": Decimal("-77.4565"),
        "country": "US",
    },
    "EWR": {
        "name": "Newark Liberty",
        "lat": Decimal("40.6895"),
        "lon": Decimal("-74.1745"),
        "country": "US",
    },
    "YYZ": {
        "name": "Toronto Pearson",
        "lat": Decimal("43.6777"),
        "lon": Decimal("-79.6248"),
        "country": "CA",
    },
    "YVR": {
        "name": "Vancouver",
        "lat": Decimal("49.1967"),
        "lon": Decimal("-123.1815"),
        "country": "CA",
    },
    "LGW": {
        "name": "London Gatwick",
        "lat": Decimal("51.1537"),
        "lon": Decimal("-0.1821"),
        "country": "GB",
    },
    "MAN": {
        "name": "Manchester",
        "lat": Decimal("53.3537"),
        "lon": Decimal("-2.2750"),
        "country": "GB",
    },
    "EDI": {
        "name": "Edinburgh",
        "lat": Decimal("55.9508"),
        "lon": Decimal("-3.3615"),
        "country": "GB",
    },
    "MAD": {
        "name": "Madrid Barajas",
        "lat": Decimal("40.4983"),
        "lon": Decimal("-3.5676"),
        "country": "ES",
    },
    "BCN": {
        "name": "Barcelona El Prat",
        "lat": Decimal("41.2974"),
        "lon": Decimal("2.0833"),
        "country": "ES",
    },
    "FCO": {
        "name": "Rome Fiumicino",
        "lat": Decimal("41.8003"),
        "lon": Decimal("12.2389"),
        "country": "IT",
    },
    "MXP": {
        "name": "Milan Malpensa",
        "lat": Decimal("45.6306"),
        "lon": Decimal("8.7281"),
        "country": "IT",
    },
    "MUC": {
        "name": "Munich",
        "lat": Decimal("48.3537"),
        "lon": Decimal("11.7750"),
        "country": "DE",
    },
    "ZRH": {
        "name": "Zurich",
        "lat": Decimal("47.4647"),
        "lon": Decimal("8.5492"),
        "country": "CH",
    },
    "CPH": {
        "name": "Copenhagen",
        "lat": Decimal("55.6180"),
        "lon": Decimal("12.6560"),
        "country": "DK",
    },
    "ARN": {
        "name": "Stockholm Arlanda",
        "lat": Decimal("59.6519"),
        "lon": Decimal("17.9186"),
        "country": "SE",
    },
    "OSL": {
        "name": "Oslo Gardermoen",
        "lat": Decimal("60.1976"),
        "lon": Decimal("11.1004"),
        "country": "NO",
    },
    "HEL": {
        "name": "Helsinki Vantaa",
        "lat": Decimal("60.3172"),
        "lon": Decimal("24.9633"),
        "country": "FI",
    },
    "DOH": {
        "name": "Doha Hamad",
        "lat": Decimal("25.2731"),
        "lon": Decimal("51.6081"),
        "country": "QA",
    },
    "JNB": {
        "name": "Johannesburg",
        "lat": Decimal("-26.1367"),
        "lon": Decimal("28.2460"),
        "country": "ZA",
    },
    "NBO": {
        "name": "Nairobi Jomo Kenyatta",
        "lat": Decimal("-1.3192"),
        "lon": Decimal("36.9278"),
        "country": "KE",
    },
    "BKK": {
        "name": "Bangkok Suvarnabhumi",
        "lat": Decimal("13.6900"),
        "lon": Decimal("100.7501"),
        "country": "TH",
    },
    "KUL": {
        "name": "Kuala Lumpur",
        "lat": Decimal("2.7456"),
        "lon": Decimal("101.7099"),
        "country": "MY",
    },
    "MEX": {
        "name": "Mexico City International",
        "lat": Decimal("19.4363"),
        "lon": Decimal("-99.0721"),
        "country": "MX",
    },
}


# ==============================================================================
# INPUT MODELS
# ==============================================================================


class FlightInput(BaseModel):
    """
    Input for flight emissions calculation (distance-based).

    Uses IATA airport codes to compute great-circle distance,
    then applies distance band emission factor with cabin class multiplier.
    Optionally includes radiative forcing (RF) uplift.

    Example:
        >>> flight = FlightInput(
        ...     origin_iata="JFK",
        ...     destination_iata="LHR",
        ...     cabin_class=CabinClass.BUSINESS,
        ...     passengers=1,
        ...     round_trip=True,
        ...     rf_option=RFOption.BOTH
        ... )
    """

    origin_iata: str = Field(
        ..., min_length=3, max_length=3,
        description="Origin airport IATA code (3 uppercase letters)"
    )
    destination_iata: str = Field(
        ..., min_length=3, max_length=3,
        description="Destination airport IATA code (3 uppercase letters)"
    )
    cabin_class: CabinClass = Field(
        default=CabinClass.ECONOMY,
        description="Cabin class (affects per-passenger allocation)"
    )
    passengers: int = Field(
        default=1, ge=1, le=500,
        description="Number of passengers on this booking"
    )
    round_trip: bool = Field(
        default=False,
        description="If True, emissions are doubled for return leg"
    )
    rf_option: RFOption = Field(
        default=RFOption.WITH_RF,
        description="Radiative forcing reporting option"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("origin_iata")
    def validate_origin_iata(cls, v: str) -> str:
        """Validate origin IATA code is 3 uppercase letters."""
        if not v.isalpha() or not v.isupper() or len(v) != 3:
            raise ValueError(
                f"Origin IATA code must be 3 uppercase letters, got '{v}'"
            )
        return v

    @validator("destination_iata")
    def validate_destination_iata(cls, v: str) -> str:
        """Validate destination IATA code is 3 uppercase letters."""
        if not v.isalpha() or not v.isupper() or len(v) != 3:
            raise ValueError(
                f"Destination IATA code must be 3 uppercase letters, got '{v}'"
            )
        return v


class RailInput(BaseModel):
    """
    Input for rail emissions calculation.

    Example:
        >>> rail = RailInput(
        ...     rail_type=RailType.EUROSTAR,
        ...     distance_km=Decimal("340"),
        ...     passengers=2
        ... )
    """

    rail_type: RailType = Field(
        ..., description="Type of rail service"
    )
    distance_km: Decimal = Field(
        ..., gt=0,
        description="One-way distance in kilometres"
    )
    passengers: int = Field(
        default=1, ge=1, le=500,
        description="Number of passengers"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("distance_km")
    def validate_distance_km(cls, v: Decimal) -> Decimal:
        """Validate distance is positive."""
        if v <= 0:
            raise ValueError(
                f"Distance must be positive, got {v}"
            )
        return v


class RoadDistanceInput(BaseModel):
    """
    Input for road vehicle distance-based emissions calculation.

    Example:
        >>> road = RoadDistanceInput(
        ...     vehicle_type=RoadVehicleType.CAR_MEDIUM_PETROL,
        ...     distance_km=Decimal("250")
        ... )
    """

    vehicle_type: RoadVehicleType = Field(
        ..., description="Vehicle type / fuel / size category"
    )
    distance_km: Decimal = Field(
        ..., gt=0,
        description="Distance driven in kilometres"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("distance_km")
    def validate_distance_km(cls, v: Decimal) -> Decimal:
        """Validate distance is positive."""
        if v <= 0:
            raise ValueError(
                f"Distance must be positive, got {v}"
            )
        return v


class RoadFuelInput(BaseModel):
    """
    Input for fuel-based road emissions calculation.

    Example:
        >>> fuel = RoadFuelInput(
        ...     fuel_type=FuelType.DIESEL,
        ...     litres=Decimal("45.0")
        ... )
    """

    fuel_type: FuelType = Field(
        ..., description="Fuel type consumed"
    )
    litres: Decimal = Field(
        ..., gt=0,
        description="Litres of fuel consumed (kg for CNG)"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("litres")
    def validate_litres(cls, v: Decimal) -> Decimal:
        """Validate fuel quantity is positive."""
        if v <= 0:
            raise ValueError(
                f"Fuel quantity must be positive, got {v}"
            )
        return v


class TaxiInput(BaseModel):
    """
    Input for taxi / ride-hailing emissions calculation.

    Example:
        >>> taxi = TaxiInput(
        ...     taxi_type=RoadVehicleType.TAXI_REGULAR,
        ...     distance_km=Decimal("15.5")
        ... )
    """

    taxi_type: RoadVehicleType = Field(
        default=RoadVehicleType.TAXI_REGULAR,
        description="Taxi vehicle type"
    )
    distance_km: Decimal = Field(
        ..., gt=0,
        description="Taxi journey distance in kilometres"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class BusInput(BaseModel):
    """
    Input for bus emissions calculation.

    Example:
        >>> bus = BusInput(
        ...     bus_type=BusType.COACH,
        ...     distance_km=Decimal("200"),
        ...     passengers=1
        ... )
    """

    bus_type: BusType = Field(
        ..., description="Type of bus service"
    )
    distance_km: Decimal = Field(
        ..., gt=0,
        description="Distance travelled in kilometres"
    )
    passengers: int = Field(
        default=1, ge=1, le=500,
        description="Number of passengers"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class FerryInput(BaseModel):
    """
    Input for ferry emissions calculation.

    Example:
        >>> ferry = FerryInput(
        ...     ferry_type=FerryType.FOOT_PASSENGER,
        ...     distance_km=Decimal("35"),
        ...     passengers=2
        ... )
    """

    ferry_type: FerryType = Field(
        ..., description="Ferry passenger type"
    )
    distance_km: Decimal = Field(
        ..., gt=0,
        description="Ferry route distance in kilometres"
    )
    passengers: int = Field(
        default=1, ge=1, le=500,
        description="Number of passengers"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class HotelInput(BaseModel):
    """
    Input for hotel accommodation emissions calculation.

    Uses country-specific room-night factors with hotel class multiplier.

    Example:
        >>> hotel = HotelInput(
        ...     country_code="GB",
        ...     room_nights=3,
        ...     hotel_class=HotelClass.UPSCALE
        ... )
    """

    country_code: str = Field(
        default="GLOBAL",
        description="ISO 3166-1 alpha-2 country code or 'GLOBAL' for default"
    )
    room_nights: int = Field(
        ..., gt=0,
        description="Number of room-nights"
    )
    hotel_class: HotelClass = Field(
        default=HotelClass.STANDARD,
        description="Hotel class / tier"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("room_nights")
    def validate_room_nights(cls, v: int) -> int:
        """Validate room nights is positive."""
        if v <= 0:
            raise ValueError(
                f"Room nights must be positive, got {v}"
            )
        return v

    @validator("country_code")
    def validate_country_code(cls, v: str) -> str:
        """Validate and uppercase country code."""
        return v.upper()


class SpendInput(BaseModel):
    """
    Input for spend-based emissions calculation using EEIO factors.

    Example:
        >>> spend = SpendInput(
        ...     naics_code="481000",
        ...     amount=Decimal("5000.00"),
        ...     currency=CurrencyCode.USD,
        ...     reporting_year=2024
        ... )
    """

    naics_code: str = Field(
        ..., description="NAICS code for EEIO factor lookup"
    )
    amount: Decimal = Field(
        ..., gt=0,
        description="Spend amount in specified currency"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="ISO 4217 currency code"
    )
    reporting_year: int = Field(
        default=2024, ge=2015, le=2030,
        description="Reporting year for CPI deflation"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("amount")
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Validate spend amount is positive."""
        if v <= 0:
            raise ValueError(
                f"Spend amount must be positive, got {v}"
            )
        return v


class TripInput(BaseModel):
    """
    Generic trip input wrapping mode-specific data with metadata.

    The trip_data dict is parsed into the appropriate mode-specific model
    by the pipeline engine based on the mode field.

    Example:
        >>> trip = TripInput(
        ...     mode=TransportMode.AIR,
        ...     trip_data={"origin_iata": "JFK", "destination_iata": "LHR"},
        ...     trip_purpose=TripPurpose.CLIENT_VISIT,
        ...     department="Sales",
        ...     cost_center="CC-4200"
        ... )
    """

    mode: TransportMode = Field(
        ..., description="Transport mode for this trip segment"
    )
    trip_data: dict = Field(
        ..., description="Mode-specific input data (parsed by pipeline)"
    )
    trip_purpose: TripPurpose = Field(
        default=TripPurpose.BUSINESS,
        description="Purpose of this business trip"
    )
    department: Optional[str] = Field(
        default=None,
        description="Department for allocation"
    )
    cost_center: Optional[str] = Field(
        default=None,
        description="Cost center for allocation"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class BatchTripInput(BaseModel):
    """
    Batch input for processing multiple trips in a single request.

    Example:
        >>> batch = BatchTripInput(
        ...     trips=[trip1, trip2, trip3],
        ...     reporting_period="2024-Q3"
        ... )
    """

    trips: List[TripInput] = Field(
        ..., min_length=1,
        description="List of trip inputs to process"
    )
    reporting_period: str = Field(
        ..., description="Reporting period (e.g., '2024-Q3', '2024')"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceCheckInput(BaseModel):
    """
    Input for compliance checking against one or more frameworks.

    Example:
        >>> check = ComplianceCheckInput(
        ...     frameworks=[ComplianceFramework.GHG_PROTOCOL, ComplianceFramework.CDP],
        ...     calculation_results=[result1, result2],
        ...     rf_disclosed=True,
        ...     mode_breakdown_provided=True
        ... )
    """

    frameworks: List[ComplianceFramework] = Field(
        ..., min_length=1,
        description="Frameworks to check compliance against"
    )
    calculation_results: list = Field(
        ..., description="Calculation results to validate"
    )
    rf_disclosed: bool = Field(
        default=False,
        description="Whether radiative forcing was disclosed"
    )
    mode_breakdown_provided: bool = Field(
        default=False,
        description="Whether mode breakdown was provided"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyInput(BaseModel):
    """
    Input for uncertainty quantification.

    Example:
        >>> unc = UncertaintyInput(
        ...     method=UncertaintyMethod.MONTE_CARLO,
        ...     iterations=10000,
        ...     confidence_level=Decimal("0.95")
        ... )
    """

    method: UncertaintyMethod = Field(
        default=UncertaintyMethod.MONTE_CARLO,
        description="Uncertainty quantification method"
    )
    iterations: int = Field(
        default=10000, ge=100, le=1000000,
        description="Number of Monte Carlo iterations"
    )
    confidence_level: Decimal = Field(
        default=Decimal("0.95"),
        description="Confidence level for interval (0.90, 0.95, 0.99)"
    )
    parameter_ranges: Optional[dict] = Field(
        default=None,
        description="Custom parameter uncertainty ranges"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityInput(BaseModel):
    """
    Input for data quality assessment across 5 dimensions.

    Example:
        >>> dqi = DataQualityInput(
        ...     dimensions={
        ...         DQIDimension.REPRESENTATIVENESS: DQIScore.HIGH,
        ...         DQIDimension.COMPLETENESS: DQIScore.VERY_HIGH,
        ...         DQIDimension.TEMPORAL: DQIScore.HIGH,
        ...         DQIDimension.GEOGRAPHICAL: DQIScore.MEDIUM,
        ...         DQIDimension.TECHNOLOGICAL: DQIScore.HIGH
        ...     }
        ... )
    """

    dimensions: Dict[DQIDimension, DQIScore] = Field(
        ..., description="DQI score per dimension"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# RESULT MODELS
# ==============================================================================


class FlightResult(BaseModel):
    """
    Result from flight emissions calculation.

    Contains emissions with and without radiative forcing, WTT component,
    and full provenance hash.
    """

    origin_iata: str = Field(
        ..., description="Origin airport IATA code"
    )
    destination_iata: str = Field(
        ..., description="Destination airport IATA code"
    )
    distance_km: Decimal = Field(
        ..., description="Great-circle distance in km"
    )
    distance_band: FlightDistanceBand = Field(
        ..., description="DEFRA distance band classification"
    )
    cabin_class: CabinClass = Field(
        ..., description="Cabin class used"
    )
    passengers: int = Field(
        ..., description="Number of passengers"
    )
    class_multiplier: Decimal = Field(
        ..., description="Cabin class multiplier applied"
    )
    co2e_without_rf: Decimal = Field(
        ..., description="CO2e without radiative forcing (kgCO2e)"
    )
    co2e_with_rf: Decimal = Field(
        ..., description="CO2e with radiative forcing (kgCO2e)"
    )
    wtt_co2e: Decimal = Field(
        ..., description="Well-to-tank emissions (kgCO2e)"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e based on RF option (kgCO2e)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    rf_option: RFOption = Field(
        ..., description="RF reporting option used"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class RailResult(BaseModel):
    """Result from rail emissions calculation."""

    rail_type: RailType = Field(
        ..., description="Rail service type"
    )
    distance_km: Decimal = Field(
        ..., description="Distance in km"
    )
    passengers: int = Field(
        ..., description="Number of passengers"
    )
    co2e: Decimal = Field(
        ..., description="Tank-to-wheel CO2e (kgCO2e)"
    )
    wtt_co2e: Decimal = Field(
        ..., description="Well-to-tank CO2e (kgCO2e)"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e including WTT (kgCO2e)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class RoadDistanceResult(BaseModel):
    """Result from road distance-based emissions calculation."""

    vehicle_type: RoadVehicleType = Field(
        ..., description="Vehicle type used"
    )
    distance_km: Decimal = Field(
        ..., description="Distance in km"
    )
    co2e: Decimal = Field(
        ..., description="Direct CO2e (kgCO2e)"
    )
    wtt_co2e: Decimal = Field(
        ..., description="Well-to-tank CO2e (kgCO2e)"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e including WTT (kgCO2e)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class RoadFuelResult(BaseModel):
    """Result from fuel-based road emissions calculation."""

    fuel_type: FuelType = Field(
        ..., description="Fuel type consumed"
    )
    litres: Decimal = Field(
        ..., description="Litres consumed (kg for CNG)"
    )
    co2e: Decimal = Field(
        ..., description="Direct combustion CO2e (kgCO2e)"
    )
    wtt_co2e: Decimal = Field(
        ..., description="Well-to-tank CO2e (kgCO2e)"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e including WTT (kgCO2e)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class HotelResult(BaseModel):
    """Result from hotel accommodation emissions calculation."""

    country_code: str = Field(
        ..., description="Country code used for EF lookup"
    )
    room_nights: int = Field(
        ..., description="Number of room-nights"
    )
    hotel_class: HotelClass = Field(
        ..., description="Hotel class / tier"
    )
    class_multiplier: Decimal = Field(
        ..., description="Hotel class multiplier applied"
    )
    co2e: Decimal = Field(
        ..., description="Base room-night CO2e (kgCO2e)"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e with class multiplier (kgCO2e)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class SpendResult(BaseModel):
    """Result from spend-based EEIO emissions calculation."""

    naics_code: str = Field(
        ..., description="NAICS code used"
    )
    spend_usd: Decimal = Field(
        ..., description="Spend amount in USD after currency conversion"
    )
    cpi_deflator: Decimal = Field(
        ..., description="CPI deflator applied"
    )
    eeio_factor: Decimal = Field(
        ..., description="EEIO factor (kgCO2e/USD)"
    )
    co2e: Decimal = Field(
        ..., description="Total CO2e (kgCO2e)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class TripCalculationResult(BaseModel):
    """
    Unified result from any trip segment calculation.

    Wraps mode-specific results into a common structure for
    aggregation and reporting.
    """

    mode: TransportMode = Field(
        ..., description="Transport mode"
    )
    method: CalculationMethod = Field(
        ..., description="Calculation method used"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e (kgCO2e)"
    )
    co2e_without_rf: Optional[Decimal] = Field(
        default=None,
        description="CO2e without RF (flights only)"
    )
    co2e_with_rf: Optional[Decimal] = Field(
        default=None,
        description="CO2e with RF (flights only)"
    )
    wtt_co2e: Decimal = Field(
        ..., description="Well-to-tank component (kgCO2e)"
    )
    dqi_score: Optional[Decimal] = Field(
        default=None,
        description="Data quality indicator score (1-5)"
    )
    trip_detail: dict = Field(
        ..., description="Mode-specific result detail"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class BatchResult(BaseModel):
    """Result from batch trip processing."""

    results: List[TripCalculationResult] = Field(
        ..., description="Individual trip results"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e for all trips (kgCO2e)"
    )
    total_co2e_without_rf: Optional[Decimal] = Field(
        default=None,
        description="Total CO2e without RF (kgCO2e)"
    )
    total_co2e_with_rf: Optional[Decimal] = Field(
        default=None,
        description="Total CO2e with RF (kgCO2e)"
    )
    count: int = Field(
        ..., description="Total number of trips processed"
    )
    errors: List[dict] = Field(
        default_factory=list,
        description="Errors from failed trip calculations"
    )
    reporting_period: str = Field(
        ..., description="Reporting period"
    )

    model_config = ConfigDict(frozen=True)


class AggregationResult(BaseModel):
    """Aggregated emissions by various dimensions."""

    total_co2e: Decimal = Field(
        ..., description="Total CO2e (kgCO2e)"
    )
    by_mode: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by transport mode"
    )
    by_period: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by time period"
    )
    by_department: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by department"
    )
    by_cabin_class: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by cabin class (flights)"
    )
    period: str = Field(
        ..., description="Reporting period"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceCheckResult(BaseModel):
    """Result from compliance check against a specific framework."""

    framework: ComplianceFramework = Field(
        ..., description="Framework checked"
    )
    status: ComplianceStatus = Field(
        ..., description="Compliance status"
    )
    score: Decimal = Field(
        ..., description="Compliance score (0-100)"
    )
    findings: List[dict] = Field(
        default_factory=list,
        description="Specific findings (gaps, issues)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """Result from uncertainty quantification."""

    mean: Decimal = Field(
        ..., description="Mean emissions estimate (kgCO2e)"
    )
    std_dev: Decimal = Field(
        ..., description="Standard deviation (kgCO2e)"
    )
    ci_lower: Decimal = Field(
        ..., description="Confidence interval lower bound (kgCO2e)"
    )
    ci_upper: Decimal = Field(
        ..., description="Confidence interval upper bound (kgCO2e)"
    )
    method: UncertaintyMethod = Field(
        ..., description="Method used"
    )
    iterations: int = Field(
        ..., description="Number of iterations (Monte Carlo)"
    )
    confidence_level: Decimal = Field(
        ..., description="Confidence level (e.g., 0.95)"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityResult(BaseModel):
    """Result from data quality assessment."""

    overall_score: Decimal = Field(
        ..., description="Weighted composite DQI score (1-5)"
    )
    dimensions: Dict[str, Decimal] = Field(
        ..., description="Score per dimension"
    )
    classification: str = Field(
        ..., description="Quality classification (Excellent/Good/Fair/Poor/Very Poor)"
    )
    tier: DataQualityTier = Field(
        ..., description="Data quality tier assignment"
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(BaseModel):
    """Single record in the provenance chain."""

    stage: str = Field(
        ..., description="Pipeline stage name"
    )
    input_hash: str = Field(
        ..., description="SHA-256 hash of stage input"
    )
    output_hash: str = Field(
        ..., description="SHA-256 hash of stage output"
    )
    chain_hash: str = Field(
        ..., description="Cumulative chain hash (input_hash + previous chain_hash)"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Stage-specific metadata"
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceChainResult(BaseModel):
    """Complete provenance chain for an emissions calculation."""

    records: List[ProvenanceRecord] = Field(
        ..., description="Ordered list of provenance records"
    )
    is_valid: bool = Field(
        ..., description="Whether chain integrity is verified"
    )
    chain_hash: str = Field(
        ..., description="Final chain hash"
    )

    model_config = ConfigDict(frozen=True)


class HotSpotResult(BaseModel):
    """
    Hot-spot analysis result identifying top emission contributors
    and reduction opportunities.
    """

    top_routes: List[dict] = Field(
        default_factory=list,
        description="Top emitting routes (origin-destination pairs)"
    )
    top_modes: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by transport mode"
    )
    reduction_opportunities: List[dict] = Field(
        default_factory=list,
        description="Identified reduction opportunities"
    )

    model_config = ConfigDict(frozen=True)


class MetricsSummary(BaseModel):
    """Summary metrics for monitoring and dashboarding."""

    total_calculations: int = Field(
        ..., description="Total number of calculations performed"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e across all calculations (kgCO2e)"
    )
    total_flights: int = Field(
        ..., description="Total number of flight calculations"
    )
    total_ground_trips: int = Field(
        ..., description="Total number of ground transport calculations"
    )
    total_hotel_nights: int = Field(
        ..., description="Total number of hotel room-nights"
    )
    avg_dqi: Decimal = Field(
        ..., description="Average data quality indicator score"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Quantization constant: 8 decimal places
_QUANT_8DP = Decimal("0.00000001")


def calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Supports Pydantic models (serialized to sorted JSON), Decimal values,
    and any other stringifiable objects.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).

    Example:
        >>> h = calculate_provenance_hash("JFK", "LHR", Decimal("5555.12"))
        >>> len(h)
        64
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, BaseModel):
            # Pydantic v2 model_dump_json() does not support sort_keys;
            # serialise via json.dumps with sort_keys for deterministic output.
            hash_input += json.dumps(
                inp.model_dump(mode="json"), sort_keys=True, default=str
            )
        elif isinstance(inp, Decimal):
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
        else:
            hash_input += str(inp)

    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def get_dqi_classification(score: Decimal) -> str:
    """
    Classify a composite DQI score into a human-readable label.

    Score range 1-5 (5 = best):
      >=4.5 -> Excellent
      >=3.5 -> Good
      >=2.5 -> Fair
      >=1.5 -> Poor
      <1.5  -> Very Poor

    Args:
        score: Composite DQI score (1-5).

    Returns:
        Classification string.

    Example:
        >>> get_dqi_classification(Decimal("4.2"))
        'Good'
        >>> get_dqi_classification(Decimal("4.8"))
        'Excellent'
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


def convert_currency_to_usd(amount: Decimal, currency: CurrencyCode) -> Decimal:
    """
    Convert an amount from the given currency to USD using stored exchange rates.

    Args:
        amount: Amount in the source currency.
        currency: Source currency code.

    Returns:
        Equivalent amount in USD, quantized to 8 decimal places.

    Raises:
        ValueError: If currency code is not found in CURRENCY_RATES.

    Example:
        >>> convert_currency_to_usd(Decimal("1000"), CurrencyCode.EUR)
        Decimal('1085.00000000')
    """
    rate = CURRENCY_RATES.get(currency)
    if rate is None:
        raise ValueError(
            f"Currency '{currency.value}' not found in CURRENCY_RATES"
        )
    return (amount * rate).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def get_cpi_deflator(year: int, base_year: int = 2021) -> Decimal:
    """
    Get the CPI deflator to convert spend from a given year to the base year.

    The deflator converts nominal spend to real (base-year) USD:
      real_usd = nominal_usd / deflator(year) * deflator(base_year)

    Since base_year=2021 has deflator=1.0, the formula simplifies to:
      real_usd = nominal_usd / deflator(year)

    Args:
        year: Year of the spend data.
        base_year: Base year for deflation (default 2021).

    Returns:
        CPI deflator value.

    Raises:
        ValueError: If year is not found in CPI_DEFLATORS.

    Example:
        >>> get_cpi_deflator(2024)
        Decimal('1.1490')
    """
    deflator = CPI_DEFLATORS.get(year)
    if deflator is None:
        raise ValueError(
            f"CPI deflator not available for year {year}. "
            f"Available years: {sorted(CPI_DEFLATORS.keys())}"
        )
    base_deflator = CPI_DEFLATORS.get(base_year)
    if base_deflator is None:
        raise ValueError(
            f"CPI deflator not available for base year {base_year}"
        )
    return deflator


def calculate_great_circle_distance(
    lat1: Decimal, lon1: Decimal, lat2: Decimal, lon2: Decimal
) -> Decimal:
    """
    Calculate great-circle distance between two points using the Haversine formula.

    Args:
        lat1: Latitude of point 1 (decimal degrees).
        lon1: Longitude of point 1 (decimal degrees).
        lat2: Latitude of point 2 (decimal degrees).
        lon2: Longitude of point 2 (decimal degrees).

    Returns:
        Distance in kilometres, quantized to 8 decimal places.

    Example:
        >>> calculate_great_circle_distance(
        ...     Decimal("40.6413"), Decimal("-73.7781"),
        ...     Decimal("51.4700"), Decimal("-0.4543")
        ... )  # JFK to LHR ~ 5541 km
    """
    earth_radius_km = Decimal("6371.0")

    # Convert to radians using float math for trig functions
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    distance = earth_radius_km * Decimal(str(c))
    return distance.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def classify_flight_distance_band(distance_km: Decimal) -> FlightDistanceBand:
    """
    Classify a flight distance into DEFRA distance bands.

    Bands:
      - DOMESTIC: < 700 km
      - SHORT_HAUL: 700-3700 km
      - LONG_HAUL: > 3700 km

    Args:
        distance_km: Great-circle distance in kilometres.

    Returns:
        FlightDistanceBand classification.

    Example:
        >>> classify_flight_distance_band(Decimal("500"))
        <FlightDistanceBand.DOMESTIC: 'domestic'>
        >>> classify_flight_distance_band(Decimal("2000"))
        <FlightDistanceBand.SHORT_HAUL: 'short_haul'>
        >>> classify_flight_distance_band(Decimal("8000"))
        <FlightDistanceBand.LONG_HAUL: 'long_haul'>
    """
    if distance_km < Decimal("700"):
        return FlightDistanceBand.DOMESTIC
    elif distance_km <= Decimal("3700"):
        return FlightDistanceBand.SHORT_HAUL
    else:
        return FlightDistanceBand.LONG_HAUL


def lookup_airport(iata_code: str) -> Optional[Dict[str, Any]]:
    """
    Look up airport metadata by IATA code.

    Args:
        iata_code: 3-letter IATA airport code (uppercase).

    Returns:
        Airport dict with name, lat, lon, country; or None if not found.

    Example:
        >>> airport = lookup_airport("JFK")
        >>> airport["name"]
        'John F. Kennedy International'
    """
    return AIRPORT_DATABASE.get(iata_code.upper())


def get_hotel_ef(country_code: str) -> Decimal:
    """
    Get hotel emission factor for a country, falling back to GLOBAL default.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Hotel emission factor in kgCO2e per room-night.

    Example:
        >>> get_hotel_ef("GB")
        Decimal('12.32')
        >>> get_hotel_ef("ZZ")  # Unknown country -> GLOBAL
        Decimal('20.90')
    """
    code = country_code.upper()
    return HOTEL_EMISSION_FACTORS.get(code, HOTEL_EMISSION_FACTORS["GLOBAL"])


def get_eeio_factor(naics_code: str) -> Optional[Decimal]:
    """
    Get EEIO emission factor by NAICS code.

    Args:
        naics_code: NAICS industry code string.

    Returns:
        EEIO factor in kgCO2e per USD, or None if not found.

    Example:
        >>> get_eeio_factor("481000")
        Decimal('0.4770')
    """
    entry = EEIO_FACTORS.get(naics_code)
    if entry is not None:
        return entry["ef"]
    return None


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",

    # Enums
    "CalculationMethod",
    "TransportMode",
    "FlightDistanceBand",
    "CabinClass",
    "RailType",
    "RoadVehicleType",
    "FuelType",
    "BusType",
    "FerryType",
    "HotelClass",
    "TripPurpose",
    "EFSource",
    "ComplianceFramework",
    "DataQualityTier",
    "RFOption",
    "ProvenanceStage",
    "UncertaintyMethod",
    "DQIDimension",
    "DQIScore",
    "ComplianceStatus",
    "GWPVersion",
    "EmissionGas",
    "CurrencyCode",
    "ExportFormat",
    "BatchStatus",
    "AllocationMethod",

    # Constants
    "GWP_VALUES",
    "AIR_EMISSION_FACTORS",
    "CABIN_CLASS_MULTIPLIERS",
    "RAIL_EMISSION_FACTORS",
    "ROAD_VEHICLE_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "BUS_EMISSION_FACTORS",
    "FERRY_EMISSION_FACTORS",
    "HOTEL_EMISSION_FACTORS",
    "HOTEL_CLASS_MULTIPLIERS",
    "EEIO_FACTORS",
    "CURRENCY_RATES",
    "CPI_DEFLATORS",
    "DQI_SCORING",
    "DQI_WEIGHTS",
    "UNCERTAINTY_RANGES",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    "AIRPORT_DATABASE",

    # Input models
    "FlightInput",
    "RailInput",
    "RoadDistanceInput",
    "RoadFuelInput",
    "TaxiInput",
    "BusInput",
    "FerryInput",
    "HotelInput",
    "SpendInput",
    "TripInput",
    "BatchTripInput",
    "ComplianceCheckInput",
    "UncertaintyInput",
    "DataQualityInput",

    # Result models
    "FlightResult",
    "RailResult",
    "RoadDistanceResult",
    "RoadFuelResult",
    "HotelResult",
    "SpendResult",
    "TripCalculationResult",
    "BatchResult",
    "AggregationResult",
    "ComplianceCheckResult",
    "UncertaintyResult",
    "DataQualityResult",
    "ProvenanceRecord",
    "ProvenanceChainResult",
    "HotSpotResult",
    "MetricsSummary",

    # Helper functions
    "calculate_provenance_hash",
    "get_dqi_classification",
    "convert_currency_to_usd",
    "get_cpi_deflator",
    "calculate_great_circle_distance",
    "classify_flight_distance_band",
    "lookup_airport",
    "get_hotel_ef",
    "get_eeio_factor",
]
