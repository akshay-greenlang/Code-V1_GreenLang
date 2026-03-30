"""
Upstream Leased Assets Agent Models (AGENT-MRV-021)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 8
(Upstream Leased Assets) emissions calculations.

Supports:
- 4 calculation methods (asset-specific, lessor-specific, average-data, spend-based)
- 4 asset categories (buildings, vehicles, equipment, IT assets)
- 8 building types x 5 climate zones with EUI benchmarks
- 8 vehicle types x 7 fuel types with per-km emission factors (DEFRA 2024)
- 6 equipment types with fuel consumption and load factor benchmarks
- 7 IT asset types with power ratings and PUE adjustments
- Grid emission factors for 11 countries + global default (IEA 2024)
- 26 US eGRID subregion emission factors (EPA 2024)
- 8 fuel emission factors with WTT (well-to-tank) factors (DEFRA 2024)
- 10 EEIO spend-based factors (NAICS codes for leasing/rental)
- 15 refrigerant GWP values (IPCC AR6)
- 5 allocation methods (floor area, headcount, revenue, equal share, custom)
- CPI deflation and multi-currency conversion (12 currencies)
- Data quality indicators (DQI) with 5-dimension scoring
- Uncertainty quantification (Monte Carlo, analytical, IPCC Tier 2)
- Compliance checking for 7 frameworks (GHG Protocol, ISO 14064, CSRD, CDP,
  SBTi, SB 253, GRI)
- SHA-256 provenance chain with 10-stage pipeline
- 30+ country-to-climate-zone mappings

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.upstream_leased_assets.models import (
    ...     BuildingAssetInput, BuildingType, ClimateZone, CalculationMethod,
    ... )
    >>> building = BuildingAssetInput(
    ...     asset_id="BLD-001",
    ...     building_type=BuildingType.OFFICE,
    ...     floor_area_sqm=Decimal("2500.0"),
    ...     climate_zone=ClimateZone.TEMPERATE,
    ...     country_code="GB",
    ...     lease_share=Decimal("1.0"),
    ...     method=CalculationMethod.AVERAGE_DATA,
    ... )
    >>> from greenlang.agents.mrv.upstream_leased_assets.models import calculate_provenance_hash
    >>> h = calculate_provenance_hash(building)
    >>> len(h)
    64
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from pydantic import Field, validator
from greenlang.schemas import GreenLangBase, utcnow, new_uuid

import hashlib
import json

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-008"
AGENT_COMPONENT: str = "AGENT-MRV-021"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_ula_"

# ==============================================================================
# ENUMERATIONS (22 total)
# ==============================================================================


class CalculationMethod(str, Enum):
    """Calculation method for upstream leased asset emissions per GHG Protocol."""

    ASSET_SPECIFIC = "asset_specific"  # Metered energy data per leased asset
    LESSOR_SPECIFIC = "lessor_specific"  # Primary data from lessor/landlord
    AVERAGE_DATA = "average_data"  # Benchmark EUI/energy intensity by asset type
    SPEND_BASED = "spend_based"  # Lease spend x EEIO factor


class LeaseType(str, Enum):
    """Lease classification per IFRS 16 / ASC 842."""

    OPERATING = "operating"  # Operating lease (not on balance sheet)
    FINANCE = "finance"  # Finance / capital lease (on balance sheet)


class AssetCategory(str, Enum):
    """High-level asset categories for upstream leased assets (4 categories)."""

    BUILDING = "building"  # Leased buildings and office spaces
    VEHICLE = "vehicle"  # Leased vehicle fleets
    EQUIPMENT = "equipment"  # Leased industrial / construction equipment
    IT_ASSET = "it_asset"  # Leased IT infrastructure and devices


class BuildingType(str, Enum):
    """Building types with distinct energy use intensity profiles (8 types)."""

    OFFICE = "office"  # Office / commercial office space
    RETAIL = "retail"  # Retail store / shopping center
    WAREHOUSE = "warehouse"  # Warehouse / distribution center
    INDUSTRIAL = "industrial"  # Industrial / manufacturing facility
    DATA_CENTER = "data_center"  # Data center / server farm
    HOTEL = "hotel"  # Hotel / hospitality
    HEALTHCARE = "healthcare"  # Hospital / healthcare facility
    EDUCATION = "education"  # School / university / training center


class VehicleType(str, Enum):
    """Leased vehicle types with distinct emission profiles (8 types)."""

    SMALL_CAR = "small_car"  # Small car (< 1.4L engine)
    MEDIUM_CAR = "medium_car"  # Medium car (1.4-2.0L engine)
    LARGE_CAR = "large_car"  # Large car (> 2.0L engine)
    SUV = "suv"  # Sport utility vehicle / crossover
    LIGHT_VAN = "light_van"  # Light commercial van (< 3.5t)
    HEAVY_VAN = "heavy_van"  # Heavy commercial van (> 3.5t)
    LIGHT_TRUCK = "light_truck"  # Light truck / pickup (< 7.5t)
    HEAVY_TRUCK = "heavy_truck"  # Heavy truck / HGV (> 7.5t)


class FuelType(str, Enum):
    """Fuel types for vehicle and equipment emission calculations (7 types)."""

    PETROL = "petrol"  # Gasoline / petrol
    DIESEL = "diesel"  # Diesel
    HYBRID = "hybrid"  # Hybrid electric vehicle (HEV)
    BEV = "bev"  # Battery electric vehicle (BEV)
    LPG = "lpg"  # Liquefied petroleum gas
    CNG = "cng"  # Compressed natural gas
    HYDROGEN = "hydrogen"  # Hydrogen fuel cell


class EquipmentType(str, Enum):
    """Leased equipment types with energy consumption benchmarks (6 types)."""

    MANUFACTURING = "manufacturing"  # Manufacturing / production machinery
    CONSTRUCTION = "construction"  # Construction equipment (excavator, crane, etc.)
    GENERATOR = "generator"  # Diesel / gas generator set
    AGRICULTURAL = "agricultural"  # Agricultural equipment (tractor, combine, etc.)
    MINING = "mining"  # Mining equipment (drill, loader, etc.)
    HVAC = "hvac"  # HVAC / refrigeration equipment


class ITAssetType(str, Enum):
    """Leased IT asset types with power consumption profiles (7 types)."""

    SERVER = "server"  # Rack server / blade server
    NETWORK_SWITCH = "network_switch"  # Network switch / router
    STORAGE = "storage"  # Storage array / SAN
    DESKTOP = "desktop"  # Desktop computer / workstation
    LAPTOP = "laptop"  # Laptop computer
    PRINTER = "printer"  # Printer / MFP
    COPIER = "copier"  # Copier / large-format printer


class EnergySource(str, Enum):
    """Energy sources consumed by leased assets (9 sources)."""

    ELECTRICITY = "electricity"  # Grid electricity
    NATURAL_GAS = "natural_gas"  # Natural gas (piped)
    HEATING_OIL = "heating_oil"  # Heating oil / fuel oil
    LPG = "lpg"  # Liquefied petroleum gas
    COAL = "coal"  # Coal / solid fuel
    DISTRICT_HEATING = "district_heating"  # District heating network
    DISTRICT_COOLING = "district_cooling"  # District cooling network
    WOOD_PELLETS = "wood_pellets"  # Wood pellets / biomass
    ON_SITE_SOLAR = "on_site_solar"  # On-site solar PV (zero operational)


class AllocationMethod(str, Enum):
    """Emissions allocation method for shared leased assets (5 methods)."""

    FLOOR_AREA = "floor_area"  # Based on occupied floor area (sqm)
    HEADCOUNT = "headcount"  # Based on employee headcount
    REVENUE = "revenue"  # Based on revenue share
    EQUAL_SHARE = "equal_share"  # Equal allocation across tenants
    CUSTOM = "custom"  # Custom allocation factor provided


class ClimateZone(str, Enum):
    """Climate zones affecting building EUI benchmarks (5 zones)."""

    TROPICAL = "tropical"  # Hot and humid year-round (Koppen A)
    ARID = "arid"  # Hot and dry (Koppen B)
    TEMPERATE = "temperate"  # Moderate (Koppen C)
    CONTINENTAL = "continental"  # Cold winters, warm summers (Koppen D)
    POLAR = "polar"  # Very cold year-round (Koppen E)


class EFSource(str, Enum):
    """Emission factor data sources for leased asset calculations (6 sources)."""

    DEFRA_2024 = "defra_2024"  # UK DEFRA/DESNZ conversion factors 2024
    IEA_2024 = "iea_2024"  # IEA energy statistics (grid factors)
    EPA_EGRID = "epa_egrid"  # US EPA eGRID subregion factors
    IPCC_AR6 = "ipcc_ar6"  # IPCC Sixth Assessment Report
    ENERGY_STAR = "energy_star"  # EPA Energy Star benchmarks
    CUSTOM = "custom"  # Custom / organization-specific factors


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework for compliance checks (7 frameworks)."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 Climate Change
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    SB_253 = "sb_253"  # California SB 253 (Climate Corporate Data Accountability Act)
    GRI = "gri"  # GRI 305 Emissions Standard


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges (3 tiers)."""

    MEASURED = "measured"  # Direct metering / primary data
    CALCULATED = "calculated"  # Engineering calculations / secondary data
    ESTIMATED = "estimated"  # Benchmarks / proxies / spend-based


class ProvenanceStage(str, Enum):
    """Processing pipeline stages for provenance tracking (10 stages)."""

    VALIDATE = "validate"  # Input validation
    CLASSIFY = "classify"  # Asset classification and categorization
    NORMALIZE = "normalize"  # Unit normalization (currency, area, energy)
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution
    CALCULATE = "calculate"  # Emissions calculation
    ALLOCATE = "allocate"  # Tenant / share allocation
    AGGREGATE = "aggregate"  # Portfolio aggregation
    COMPLIANCE = "compliance"  # Compliance checks
    PROVENANCE = "provenance"  # Provenance chain computation
    SEAL = "seal"  # Provenance chain sealing


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method (3 methods)."""

    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    ANALYTICAL = "analytical"  # Analytical error propagation
    IPCC_TIER2 = "ipcc_tier2"  # IPCC Tier 2 default ranges


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol (5 dimensions)."""

    RELIABILITY = "reliability"  # Verification status and data origin
    COMPLETENESS = "completeness"  # Fraction of data coverage
    TEMPORAL = "temporal"  # Temporal correlation to reporting year
    GEOGRAPHICAL = "geographical"  # Geographical correlation to activity
    TECHNOLOGICAL = "technological"  # Technological correlation to activity


class DQIScore(str, Enum):
    """Data Quality Indicator scores (5-point scale, 5 = best)."""

    VERY_GOOD = "very_good"  # 5 - Metered / verified primary data
    GOOD = "good"  # 4 - Lessor-reported / audited data
    FAIR = "fair"  # 3 - Benchmark / industry average data
    POOR = "poor"  # 2 - Estimated / proxy data
    VERY_POOR = "very_poor"  # 1 - Spend-based / generic data


class ComplianceStatus(str, Enum):
    """Compliance check result status (3 statuses)."""

    PASS_ = "pass"  # Fully compliant (pass_ avoids Python keyword)
    FAIL = "fail"  # Non-compliant
    WARNING = "warning"  # Partially compliant / needs attention


class GWPVersion(str, Enum):
    """IPCC Global Warming Potential assessment report version (4 versions)."""

    AR4 = "ar4"  # Fourth Assessment Report (100-year)
    AR5 = "ar5"  # Fifth Assessment Report (100-year)
    AR6 = "ar6"  # Sixth Assessment Report (100-year)
    AR6_20YR = "ar6_20yr"  # Sixth Assessment Report (20-year)


class EmissionGas(str, Enum):
    """Greenhouse gas types relevant to leased asset emissions (3 gases)."""

    CO2 = "co2"  # Carbon dioxide
    CH4 = "ch4"  # Methane
    N2O = "n2o"  # Nitrous oxide


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for spend-based calculations (12 currencies)."""

    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    CAD = "CAD"  # Canadian Dollar
    AUD = "AUD"  # Australian Dollar
    JPY = "JPY"  # Japanese Yen
    CNY = "CNY"  # Chinese Yuan
    INR = "INR"  # Indian Rupee
    CHF = "CHF"  # Swiss Franc
    BRL = "BRL"  # Brazilian Real
    ZAR = "ZAR"  # South African Rand
    KRW = "KRW"  # South Korean Won


# ==============================================================================
# CONSTANT TABLES (14 total)
# ==============================================================================

# Quantization constant: 8 decimal places
_QUANT_8DP = Decimal("0.00000001")


def _q(value: str) -> Decimal:
    """Quantize a string to 8 decimal places for consistent precision."""
    return Decimal(value).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------
# 1. Building EUI Benchmarks (kWh/m2/year) by building type and climate zone
#    Source: Energy Star, CIBSE TM46, IEA
#    Keys: (BuildingType, ClimateZone) -> {eui_kwh_sqm, gas_fraction, source}
#    gas_fraction = fraction of total energy from natural gas (remainder = electricity)
# --------------------------------------------------------------------------
BUILDING_EUI_BENCHMARKS: Dict[
    Tuple[str, str], Dict[str, Any]
] = {
    # OFFICE - 5 climate zones
    ("office", "continental"): {
        "eui_kwh_sqm": _q("180.00000000"),
        "gas_fraction": _q("0.45000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("office", "temperate"): {
        "eui_kwh_sqm": _q("150.00000000"),
        "gas_fraction": _q("0.35000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("office", "tropical"): {
        "eui_kwh_sqm": _q("210.00000000"),
        "gas_fraction": _q("0.10000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("office", "arid"): {
        "eui_kwh_sqm": _q("190.00000000"),
        "gas_fraction": _q("0.20000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("office", "polar"): {
        "eui_kwh_sqm": _q("220.00000000"),
        "gas_fraction": _q("0.55000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    # RETAIL - 5 climate zones
    ("retail", "continental"): {
        "eui_kwh_sqm": _q("220.00000000"),
        "gas_fraction": _q("0.40000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("retail", "temperate"): {
        "eui_kwh_sqm": _q("190.00000000"),
        "gas_fraction": _q("0.30000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("retail", "tropical"): {
        "eui_kwh_sqm": _q("270.00000000"),
        "gas_fraction": _q("0.08000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("retail", "arid"): {
        "eui_kwh_sqm": _q("240.00000000"),
        "gas_fraction": _q("0.18000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("retail", "polar"): {
        "eui_kwh_sqm": _q("260.00000000"),
        "gas_fraction": _q("0.50000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    # WAREHOUSE - 5 climate zones
    ("warehouse", "continental"): {
        "eui_kwh_sqm": _q("80.00000000"),
        "gas_fraction": _q("0.50000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("warehouse", "temperate"): {
        "eui_kwh_sqm": _q("65.00000000"),
        "gas_fraction": _q("0.40000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("warehouse", "tropical"): {
        "eui_kwh_sqm": _q("100.00000000"),
        "gas_fraction": _q("0.10000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("warehouse", "arid"): {
        "eui_kwh_sqm": _q("85.00000000"),
        "gas_fraction": _q("0.25000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("warehouse", "polar"): {
        "eui_kwh_sqm": _q("110.00000000"),
        "gas_fraction": _q("0.60000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    # INDUSTRIAL - 5 climate zones
    ("industrial", "continental"): {
        "eui_kwh_sqm": _q("200.00000000"),
        "gas_fraction": _q("0.45000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("industrial", "temperate"): {
        "eui_kwh_sqm": _q("170.00000000"),
        "gas_fraction": _q("0.38000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("industrial", "tropical"): {
        "eui_kwh_sqm": _q("240.00000000"),
        "gas_fraction": _q("0.12000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("industrial", "arid"): {
        "eui_kwh_sqm": _q("210.00000000"),
        "gas_fraction": _q("0.22000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("industrial", "polar"): {
        "eui_kwh_sqm": _q("250.00000000"),
        "gas_fraction": _q("0.55000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    # DATA_CENTER - 5 climate zones
    ("data_center", "continental"): {
        "eui_kwh_sqm": _q("800.00000000"),
        "gas_fraction": _q("0.02000000"),
        "source": "Energy Star / Uptime Institute",
    },
    ("data_center", "temperate"): {
        "eui_kwh_sqm": _q("800.00000000"),
        "gas_fraction": _q("0.02000000"),
        "source": "Energy Star / Uptime Institute",
    },
    ("data_center", "tropical"): {
        "eui_kwh_sqm": _q("920.00000000"),
        "gas_fraction": _q("0.01000000"),
        "source": "Energy Star / Uptime Institute",
    },
    ("data_center", "arid"): {
        "eui_kwh_sqm": _q("850.00000000"),
        "gas_fraction": _q("0.01500000"),
        "source": "Energy Star / Uptime Institute",
    },
    ("data_center", "polar"): {
        "eui_kwh_sqm": _q("750.00000000"),
        "gas_fraction": _q("0.03000000"),
        "source": "Energy Star / Uptime Institute",
    },
    # HOTEL - 5 climate zones
    ("hotel", "continental"): {
        "eui_kwh_sqm": _q("250.00000000"),
        "gas_fraction": _q("0.40000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("hotel", "temperate"): {
        "eui_kwh_sqm": _q("220.00000000"),
        "gas_fraction": _q("0.35000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("hotel", "tropical"): {
        "eui_kwh_sqm": _q("310.00000000"),
        "gas_fraction": _q("0.10000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("hotel", "arid"): {
        "eui_kwh_sqm": _q("280.00000000"),
        "gas_fraction": _q("0.20000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("hotel", "polar"): {
        "eui_kwh_sqm": _q("300.00000000"),
        "gas_fraction": _q("0.50000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    # HEALTHCARE - 5 climate zones
    ("healthcare", "continental"): {
        "eui_kwh_sqm": _q("350.00000000"),
        "gas_fraction": _q("0.40000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("healthcare", "temperate"): {
        "eui_kwh_sqm": _q("300.00000000"),
        "gas_fraction": _q("0.35000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("healthcare", "tropical"): {
        "eui_kwh_sqm": _q("410.00000000"),
        "gas_fraction": _q("0.12000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("healthcare", "arid"): {
        "eui_kwh_sqm": _q("370.00000000"),
        "gas_fraction": _q("0.22000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("healthcare", "polar"): {
        "eui_kwh_sqm": _q("420.00000000"),
        "gas_fraction": _q("0.50000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    # EDUCATION - 5 climate zones
    ("education", "continental"): {
        "eui_kwh_sqm": _q("140.00000000"),
        "gas_fraction": _q("0.45000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("education", "temperate"): {
        "eui_kwh_sqm": _q("120.00000000"),
        "gas_fraction": _q("0.35000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("education", "tropical"): {
        "eui_kwh_sqm": _q("170.00000000"),
        "gas_fraction": _q("0.10000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("education", "arid"): {
        "eui_kwh_sqm": _q("150.00000000"),
        "gas_fraction": _q("0.20000000"),
        "source": "Energy Star / CIBSE TM46",
    },
    ("education", "polar"): {
        "eui_kwh_sqm": _q("180.00000000"),
        "gas_fraction": _q("0.55000000"),
        "source": "Energy Star / CIBSE TM46",
    },
}

# --------------------------------------------------------------------------
# 2. Grid Emission Factors (kgCO2e per kWh) by country - IEA 2024
#    Keys: country code string -> {ef_per_kwh, wtt_per_kwh, source}
# --------------------------------------------------------------------------
GRID_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "US": {
        "ef_per_kwh": _q("0.37170000"),
        "wtt_per_kwh": _q("0.04851000"),
        "source": "IEA 2024",
    },
    "GB": {
        "ef_per_kwh": _q("0.20707000"),
        "wtt_per_kwh": _q("0.02703000"),
        "source": "IEA 2024",
    },
    "DE": {
        "ef_per_kwh": _q("0.33800000"),
        "wtt_per_kwh": _q("0.04412000"),
        "source": "IEA 2024",
    },
    "FR": {
        "ef_per_kwh": _q("0.05100000"),
        "wtt_per_kwh": _q("0.00666000"),
        "source": "IEA 2024",
    },
    "JP": {
        "ef_per_kwh": _q("0.43400000"),
        "wtt_per_kwh": _q("0.05664000"),
        "source": "IEA 2024",
    },
    "CN": {
        "ef_per_kwh": _q("0.55600000"),
        "wtt_per_kwh": _q("0.07256000"),
        "source": "IEA 2024",
    },
    "IN": {
        "ef_per_kwh": _q("0.70800000"),
        "wtt_per_kwh": _q("0.09240000"),
        "source": "IEA 2024",
    },
    "AU": {
        "ef_per_kwh": _q("0.65600000"),
        "wtt_per_kwh": _q("0.08562000"),
        "source": "IEA 2024",
    },
    "CA": {
        "ef_per_kwh": _q("0.12000000"),
        "wtt_per_kwh": _q("0.01566000"),
        "source": "IEA 2024",
    },
    "BR": {
        "ef_per_kwh": _q("0.07400000"),
        "wtt_per_kwh": _q("0.00966000"),
        "source": "IEA 2024",
    },
    "SE": {
        "ef_per_kwh": _q("0.00800000"),
        "wtt_per_kwh": _q("0.00104000"),
        "source": "IEA 2024",
    },
    "GLOBAL": {
        "ef_per_kwh": _q("0.43600000"),
        "wtt_per_kwh": _q("0.05690000"),
        "source": "IEA 2024",
    },
}

# --------------------------------------------------------------------------
# 3. US eGRID Subregion Emission Factors (kgCO2e per kWh) - EPA 2024
#    Keys: eGRID subregion acronym -> ef_per_kwh
# --------------------------------------------------------------------------
US_EGRID_FACTORS: Dict[str, Decimal] = {
    "CAMX": _q("0.22800000"),
    "ERCT": _q("0.37800000"),
    "FRCC": _q("0.36400000"),
    "MROE": _q("0.58200000"),
    "MROW": _q("0.41600000"),
    "NEWE": _q("0.21300000"),
    "NWPP": _q("0.28600000"),
    "NYCW": _q("0.22100000"),
    "NYLI": _q("0.30500000"),
    "NYUP": _q("0.12300000"),
    "RFCE": _q("0.30100000"),
    "RFCM": _q("0.53800000"),
    "RFCW": _q("0.44100000"),
    "RMPA": _q("0.58900000"),
    "SPNO": _q("0.49500000"),
    "SPSO": _q("0.42700000"),
    "SRMV": _q("0.35800000"),
    "SRMW": _q("0.62200000"),
    "SRSO": _q("0.38100000"),
    "SRTV": _q("0.41500000"),
    "SRVC": _q("0.30300000"),
    "AZNM": _q("0.38600000"),
    "HIMS": _q("0.49200000"),
    "HIOA": _q("0.61500000"),
    "AKGD": _q("0.36100000"),
    "AKMS": _q("0.19200000"),
}

# --------------------------------------------------------------------------
# 4. Fuel Emission Factors (kgCO2e) - DEFRA 2024
#    Keys: fuel name -> {ef_per_kwh, ef_per_litre, wtt_per_kwh, wtt_per_litre}
#    Not all fuels have both per-kWh and per-litre; missing keys are absent.
# --------------------------------------------------------------------------
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "ef_per_kwh": _q("0.18316000"),
        "wtt_per_kwh": _q("0.02391000"),
    },
    "heating_oil": {
        "ef_per_kwh": _q("0.24674000"),
        "wtt_per_kwh": _q("0.05757000"),
    },
    "lpg": {
        "ef_per_kwh": _q("0.21449000"),
        "wtt_per_kwh": _q("0.03252000"),
    },
    "coal": {
        "ef_per_kwh": _q("0.32390000"),
        "wtt_per_kwh": _q("0.03923000"),
    },
    "wood_pellets": {
        "ef_per_kwh": _q("0.01553000"),
        "wtt_per_kwh": _q("0.01264000"),
    },
    "district_heating": {
        "ef_per_kwh": _q("0.16200000"),
        "wtt_per_kwh": _q("0.02600000"),
    },
    "district_cooling": {
        "ef_per_kwh": _q("0.07100000"),
        "wtt_per_kwh": _q("0.01100000"),
    },
    "petrol": {
        "ef_per_litre": _q("2.31480000"),
        "wtt_per_litre": _q("0.58549000"),
    },
    "diesel": {
        "ef_per_litre": _q("2.70370000"),
        "wtt_per_litre": _q("0.60927000"),
    },
}

# --------------------------------------------------------------------------
# 5. Vehicle Emission Factors (kgCO2e per km) - DEFRA 2024
#    Keys: (VehicleType, FuelType) -> {ef_per_km, wtt_per_km, source}
#    Only applicable (vehicle_type, fuel_type) combinations are populated.
# --------------------------------------------------------------------------
VEHICLE_EMISSION_FACTORS: Dict[
    Tuple[str, str], Dict[str, Any]
] = {
    # SMALL_CAR
    ("small_car", "petrol"): {
        "ef_per_km": _q("0.14920000"),
        "wtt_per_km": _q("0.03770000"),
        "source": "DEFRA 2024",
    },
    ("small_car", "diesel"): {
        "ef_per_km": _q("0.13860000"),
        "wtt_per_km": _q("0.01964000"),
        "source": "DEFRA 2024",
    },
    ("small_car", "hybrid"): {
        "ef_per_km": _q("0.10450000"),
        "wtt_per_km": _q("0.02641000"),
        "source": "DEFRA 2024",
    },
    ("small_car", "bev"): {
        "ef_per_km": _q("0.04600000"),
        "wtt_per_km": _q("0.00972000"),
        "source": "DEFRA 2024",
    },
    # MEDIUM_CAR
    ("medium_car", "petrol"): {
        "ef_per_km": _q("0.18380000"),
        "wtt_per_km": _q("0.04645000"),
        "source": "DEFRA 2024",
    },
    ("medium_car", "diesel"): {
        "ef_per_km": _q("0.16720000"),
        "wtt_per_km": _q("0.02370000"),
        "source": "DEFRA 2024",
    },
    ("medium_car", "hybrid"): {
        "ef_per_km": _q("0.12870000"),
        "wtt_per_km": _q("0.03252000"),
        "source": "DEFRA 2024",
    },
    ("medium_car", "bev"): {
        "ef_per_km": _q("0.05300000"),
        "wtt_per_km": _q("0.01120000"),
        "source": "DEFRA 2024",
    },
    # LARGE_CAR
    ("large_car", "petrol"): {
        "ef_per_km": _q("0.27980000"),
        "wtt_per_km": _q("0.07073000"),
        "source": "DEFRA 2024",
    },
    ("large_car", "diesel"): {
        "ef_per_km": _q("0.21260000"),
        "wtt_per_km": _q("0.03013000"),
        "source": "DEFRA 2024",
    },
    ("large_car", "hybrid"): {
        "ef_per_km": _q("0.17620000"),
        "wtt_per_km": _q("0.04453000"),
        "source": "DEFRA 2024",
    },
    ("large_car", "bev"): {
        "ef_per_km": _q("0.07200000"),
        "wtt_per_km": _q("0.01521000"),
        "source": "DEFRA 2024",
    },
    # SUV
    ("suv", "petrol"): {
        "ef_per_km": _q("0.23140000"),
        "wtt_per_km": _q("0.05849000"),
        "source": "DEFRA 2024",
    },
    ("suv", "diesel"): {
        "ef_per_km": _q("0.19860000"),
        "wtt_per_km": _q("0.02815000"),
        "source": "DEFRA 2024",
    },
    ("suv", "hybrid"): {
        "ef_per_km": _q("0.15430000"),
        "wtt_per_km": _q("0.03899000"),
        "source": "DEFRA 2024",
    },
    ("suv", "bev"): {
        "ef_per_km": _q("0.06500000"),
        "wtt_per_km": _q("0.01373000"),
        "source": "DEFRA 2024",
    },
    # LIGHT_VAN
    ("light_van", "petrol"): {
        "ef_per_km": _q("0.20840000"),
        "wtt_per_km": _q("0.05267000"),
        "source": "DEFRA 2024",
    },
    ("light_van", "diesel"): {
        "ef_per_km": _q("0.24190000"),
        "wtt_per_km": _q("0.03429000"),
        "source": "DEFRA 2024",
    },
    ("light_van", "hybrid"): {
        "ef_per_km": _q("0.18500000"),
        "wtt_per_km": _q("0.04676000"),
        "source": "DEFRA 2024",
    },
    ("light_van", "bev"): {
        "ef_per_km": _q("0.08100000"),
        "wtt_per_km": _q("0.01711000"),
        "source": "DEFRA 2024",
    },
    # HEAVY_VAN
    ("heavy_van", "petrol"): {
        "ef_per_km": _q("0.26330000"),
        "wtt_per_km": _q("0.06654000"),
        "source": "DEFRA 2024",
    },
    ("heavy_van", "diesel"): {
        "ef_per_km": _q("0.30590000"),
        "wtt_per_km": _q("0.04336000"),
        "source": "DEFRA 2024",
    },
    ("heavy_van", "hybrid"): {
        "ef_per_km": _q("0.22100000"),
        "wtt_per_km": _q("0.05586000"),
        "source": "DEFRA 2024",
    },
    ("heavy_van", "bev"): {
        "ef_per_km": _q("0.10200000"),
        "wtt_per_km": _q("0.02155000"),
        "source": "DEFRA 2024",
    },
    # LIGHT_TRUCK
    ("light_truck", "diesel"): {
        "ef_per_km": _q("0.46210000"),
        "wtt_per_km": _q("0.06551000"),
        "source": "DEFRA 2024",
    },
    ("light_truck", "petrol"): {
        "ef_per_km": _q("0.39860000"),
        "wtt_per_km": _q("0.10076000"),
        "source": "DEFRA 2024",
    },
    ("light_truck", "cng"): {
        "ef_per_km": _q("0.41500000"),
        "wtt_per_km": _q("0.05418000"),
        "source": "DEFRA 2024",
    },
    ("light_truck", "bev"): {
        "ef_per_km": _q("0.15600000"),
        "wtt_per_km": _q("0.03295000"),
        "source": "DEFRA 2024",
    },
    # HEAVY_TRUCK
    ("heavy_truck", "diesel"): {
        "ef_per_km": _q("0.85740000"),
        "wtt_per_km": _q("0.12152000"),
        "source": "DEFRA 2024",
    },
    ("heavy_truck", "petrol"): {
        "ef_per_km": _q("0.76120000"),
        "wtt_per_km": _q("0.19237000"),
        "source": "DEFRA 2024",
    },
    ("heavy_truck", "cng"): {
        "ef_per_km": _q("0.72800000"),
        "wtt_per_km": _q("0.09502000"),
        "source": "DEFRA 2024",
    },
    ("heavy_truck", "hydrogen"): {
        "ef_per_km": _q("0.00000000"),
        "wtt_per_km": _q("0.42500000"),
        "source": "DEFRA 2024",
    },
}

# --------------------------------------------------------------------------
# 6. Equipment Benchmarks - 6 equipment types
#    Keys: EquipmentType value -> {default_hours, load_factor, fuel_type,
#           fuel_consumption_l_per_hr, source}
# --------------------------------------------------------------------------
EQUIPMENT_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "default_hours": _q("2000.00000000"),
        "load_factor": _q("0.65000000"),
        "fuel_type": "electricity",
        "fuel_consumption_l_per_hr": _q("0.00000000"),
        "electricity_kw": _q("75.00000000"),
        "source": "Industry benchmark / IEA",
    },
    "construction": {
        "default_hours": _q("1500.00000000"),
        "load_factor": _q("0.55000000"),
        "fuel_type": "diesel",
        "fuel_consumption_l_per_hr": _q("18.50000000"),
        "electricity_kw": _q("0.00000000"),
        "source": "EPA NONROAD / DEFRA 2024",
    },
    "generator": {
        "default_hours": _q("1000.00000000"),
        "load_factor": _q("0.70000000"),
        "fuel_type": "diesel",
        "fuel_consumption_l_per_hr": _q("12.00000000"),
        "electricity_kw": _q("0.00000000"),
        "source": "EPA NONROAD / DEFRA 2024",
    },
    "agricultural": {
        "default_hours": _q("1200.00000000"),
        "load_factor": _q("0.50000000"),
        "fuel_type": "diesel",
        "fuel_consumption_l_per_hr": _q("15.00000000"),
        "electricity_kw": _q("0.00000000"),
        "source": "EPA NONROAD / DEFRA 2024",
    },
    "mining": {
        "default_hours": _q("2500.00000000"),
        "load_factor": _q("0.60000000"),
        "fuel_type": "diesel",
        "fuel_consumption_l_per_hr": _q("25.00000000"),
        "electricity_kw": _q("0.00000000"),
        "source": "EPA NONROAD / DEFRA 2024",
    },
    "hvac": {
        "default_hours": _q("3000.00000000"),
        "load_factor": _q("0.45000000"),
        "fuel_type": "electricity",
        "fuel_consumption_l_per_hr": _q("0.00000000"),
        "electricity_kw": _q("15.00000000"),
        "source": "ASHRAE / Energy Star",
    },
}

# --------------------------------------------------------------------------
# 7. IT Power Ratings - 7 IT asset types
#    Keys: ITAssetType value -> {typical_power_kw, utilization, pue_default,
#           annual_hours, source}
# --------------------------------------------------------------------------
IT_POWER_RATINGS: Dict[str, Dict[str, Any]] = {
    "server": {
        "typical_power_kw": _q("0.50000000"),
        "utilization": _q("0.65000000"),
        "pue_default": _q("1.58000000"),
        "annual_hours": _q("8760.00000000"),
        "source": "Uptime Institute / EPA Energy Star",
    },
    "network_switch": {
        "typical_power_kw": _q("0.15000000"),
        "utilization": _q("0.80000000"),
        "pue_default": _q("1.58000000"),
        "annual_hours": _q("8760.00000000"),
        "source": "Uptime Institute / EPA Energy Star",
    },
    "storage": {
        "typical_power_kw": _q("0.80000000"),
        "utilization": _q("0.50000000"),
        "pue_default": _q("1.58000000"),
        "annual_hours": _q("8760.00000000"),
        "source": "Uptime Institute / EPA Energy Star",
    },
    "desktop": {
        "typical_power_kw": _q("0.15000000"),
        "utilization": _q("0.40000000"),
        "pue_default": _q("1.00000000"),
        "annual_hours": _q("2080.00000000"),
        "source": "EPA Energy Star",
    },
    "laptop": {
        "typical_power_kw": _q("0.05000000"),
        "utilization": _q("0.50000000"),
        "pue_default": _q("1.00000000"),
        "annual_hours": _q("2080.00000000"),
        "source": "EPA Energy Star",
    },
    "printer": {
        "typical_power_kw": _q("0.35000000"),
        "utilization": _q("0.10000000"),
        "pue_default": _q("1.00000000"),
        "annual_hours": _q("2080.00000000"),
        "source": "EPA Energy Star",
    },
    "copier": {
        "typical_power_kw": _q("1.10000000"),
        "utilization": _q("0.08000000"),
        "pue_default": _q("1.00000000"),
        "annual_hours": _q("2080.00000000"),
        "source": "EPA Energy Star",
    },
}

# --------------------------------------------------------------------------
# 8. EEIO Factors for spend-based calculation (kgCO2e per USD)
#    Source: EPA USEEIO v2.0 / Exiobase 3
#    Keys: NAICS code -> {name, ef}
# --------------------------------------------------------------------------
EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "531110": {
        "name": "Lessors of residential buildings",
        "ef": _q("0.19000000"),
    },
    "531120": {
        "name": "Lessors of nonresidential buildings",
        "ef": _q("0.22000000"),
    },
    "531130": {
        "name": "Lessors of miniwarehouses and self-storage units",
        "ef": _q("0.18000000"),
    },
    "531190": {
        "name": "Lessors of other real estate property",
        "ef": _q("0.20000000"),
    },
    "532100": {
        "name": "Automotive equipment rental and leasing",
        "ef": _q("0.24000000"),
    },
    "532400": {
        "name": "Commercial/industrial machinery rental and leasing",
        "ef": _q("0.28000000"),
    },
    "532200": {
        "name": "Consumer goods rental",
        "ef": _q("0.21000000"),
    },
    "518210": {
        "name": "Data processing, hosting, and related services",
        "ef": _q("0.35000000"),
    },
    "541500": {
        "name": "Computer systems design and related services",
        "ef": _q("0.16000000"),
    },
    "238000": {
        "name": "Specialty trade contractors (building fit-out)",
        "ef": _q("0.32000000"),
    },
}

# --------------------------------------------------------------------------
# 9. Currency Exchange Rates to USD (approximate mid-market rates)
# --------------------------------------------------------------------------
CURRENCY_RATES: Dict[CurrencyCode, Decimal] = {
    CurrencyCode.USD: _q("1.00000000"),
    CurrencyCode.EUR: _q("1.08500000"),
    CurrencyCode.GBP: _q("1.26500000"),
    CurrencyCode.CAD: _q("0.74100000"),
    CurrencyCode.AUD: _q("0.65200000"),
    CurrencyCode.JPY: _q("0.00666700"),
    CurrencyCode.CNY: _q("0.13780000"),
    CurrencyCode.INR: _q("0.01198000"),
    CurrencyCode.CHF: _q("1.12800000"),
    CurrencyCode.BRL: _q("0.19900000"),
    CurrencyCode.ZAR: _q("0.05340000"),
    CurrencyCode.KRW: _q("0.00075000"),
}

# --------------------------------------------------------------------------
# 10. CPI Deflators for spend-based calculation (base year 2021 = 1.0)
#     Source: US BLS CPI-U / OECD CPI
# --------------------------------------------------------------------------
CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: _q("0.84900000"),
    2016: _q("0.85970000"),
    2017: _q("0.87810000"),
    2018: _q("0.89970000"),
    2019: _q("0.91530000"),
    2020: _q("0.92710000"),
    2021: _q("1.00000000"),
    2022: _q("1.08000000"),
    2023: _q("1.11520000"),
    2024: _q("1.14900000"),
    2025: _q("1.17800000"),
}

# --------------------------------------------------------------------------
# 11. Climate Zone Map - country code to default climate zone
#     Source: Koppen-Geiger classification (dominant zone per country)
# --------------------------------------------------------------------------
CLIMATE_ZONE_MAP: Dict[str, str] = {
    "US": "temperate",
    "GB": "temperate",
    "DE": "temperate",
    "FR": "temperate",
    "JP": "temperate",
    "CN": "continental",
    "IN": "tropical",
    "AU": "arid",
    "CA": "continental",
    "BR": "tropical",
    "SE": "continental",
    "NO": "continental",
    "FI": "continental",
    "RU": "continental",
    "KR": "continental",
    "MX": "tropical",
    "ID": "tropical",
    "TH": "tropical",
    "MY": "tropical",
    "SG": "tropical",
    "PH": "tropical",
    "VN": "tropical",
    "NG": "tropical",
    "EG": "arid",
    "SA": "arid",
    "AE": "arid",
    "IL": "arid",
    "ZA": "temperate",
    "AR": "temperate",
    "CL": "temperate",
    "NZ": "temperate",
    "IT": "temperate",
    "ES": "temperate",
    "PT": "temperate",
    "GR": "temperate",
    "TR": "temperate",
    "PL": "continental",
    "IS": "polar",
    "GL": "polar",
}

# --------------------------------------------------------------------------
# 12. Refrigerant GWPs (100-year, IPCC AR6)
#     Keys: refrigerant name -> GWP value
# --------------------------------------------------------------------------
REFRIGERANT_GWPS: Dict[str, Decimal] = {
    "R-134a": _q("1530.00000000"),
    "R-410A": _q("2088.00000000"),
    "R-32": _q("675.00000000"),
    "R-407C": _q("1774.00000000"),
    "R-22": _q("1810.00000000"),
    "R-404A": _q("3922.00000000"),
    "R-507A": _q("3985.00000000"),
    "R-290": _q("3.00000000"),
    "R-600a": _q("3.00000000"),
    "R-744": _q("1.00000000"),
    "R-1234yf": _q("4.00000000"),
    "R-1234ze": _q("7.00000000"),
    "R-717": _q("0.00000000"),
    "R-123": _q("77.00000000"),
    "R-245fa": _q("1030.00000000"),
}

# --------------------------------------------------------------------------
# 13. DQI Scoring - 5 dimensions with weights summing to 1.0
#     Scoring matrix: 1-5 scale per dimension
# --------------------------------------------------------------------------
DQI_SCORING: Dict[DQIDimension, Dict[DQIScore, Decimal]] = {
    DQIDimension.RELIABILITY: {
        DQIScore.VERY_GOOD: _q("5.00000000"),
        DQIScore.GOOD: _q("4.00000000"),
        DQIScore.FAIR: _q("3.00000000"),
        DQIScore.POOR: _q("2.00000000"),
        DQIScore.VERY_POOR: _q("1.00000000"),
    },
    DQIDimension.COMPLETENESS: {
        DQIScore.VERY_GOOD: _q("5.00000000"),
        DQIScore.GOOD: _q("4.00000000"),
        DQIScore.FAIR: _q("3.00000000"),
        DQIScore.POOR: _q("2.00000000"),
        DQIScore.VERY_POOR: _q("1.00000000"),
    },
    DQIDimension.TEMPORAL: {
        DQIScore.VERY_GOOD: _q("5.00000000"),
        DQIScore.GOOD: _q("4.00000000"),
        DQIScore.FAIR: _q("3.00000000"),
        DQIScore.POOR: _q("2.00000000"),
        DQIScore.VERY_POOR: _q("1.00000000"),
    },
    DQIDimension.GEOGRAPHICAL: {
        DQIScore.VERY_GOOD: _q("5.00000000"),
        DQIScore.GOOD: _q("4.00000000"),
        DQIScore.FAIR: _q("3.00000000"),
        DQIScore.POOR: _q("2.00000000"),
        DQIScore.VERY_POOR: _q("1.00000000"),
    },
    DQIDimension.TECHNOLOGICAL: {
        DQIScore.VERY_GOOD: _q("5.00000000"),
        DQIScore.GOOD: _q("4.00000000"),
        DQIScore.FAIR: _q("3.00000000"),
        DQIScore.POOR: _q("2.00000000"),
        DQIScore.VERY_POOR: _q("1.00000000"),
    },
}

# DQI dimension weights (sum to 1.0)
DQI_WEIGHTS: Dict[DQIDimension, Decimal] = {
    DQIDimension.RELIABILITY: _q("0.25000000"),
    DQIDimension.COMPLETENESS: _q("0.25000000"),
    DQIDimension.TEMPORAL: _q("0.20000000"),
    DQIDimension.GEOGRAPHICAL: _q("0.15000000"),
    DQIDimension.TECHNOLOGICAL: _q("0.15000000"),
}

# --------------------------------------------------------------------------
# 14. Uncertainty Ranges by calculation method and data quality tier
#     Values represent the half-width of the 95% CI as a fraction
# --------------------------------------------------------------------------
UNCERTAINTY_RANGES: Dict[str, Dict[DataQualityTier, Decimal]] = {
    "asset_specific": {
        DataQualityTier.MEASURED: _q("0.05000000"),
        DataQualityTier.CALCULATED: _q("0.15000000"),
        DataQualityTier.ESTIMATED: _q("0.30000000"),
    },
    "lessor_specific": {
        DataQualityTier.MEASURED: _q("0.10000000"),
        DataQualityTier.CALCULATED: _q("0.20000000"),
        DataQualityTier.ESTIMATED: _q("0.40000000"),
    },
    "average_data": {
        DataQualityTier.MEASURED: _q("0.20000000"),
        DataQualityTier.CALCULATED: _q("0.35000000"),
        DataQualityTier.ESTIMATED: _q("0.50000000"),
    },
    "spend_based": {
        DataQualityTier.MEASURED: _q("0.30000000"),
        DataQualityTier.CALCULATED: _q("0.50000000"),
        DataQualityTier.ESTIMATED: _q("0.70000000"),
    },
}

# Required disclosures per compliance framework for upstream leased assets (Cat 8)
FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_co2e",
        "method_used",
        "ef_sources",
        "exclusions",
        "dqi_score",
        "asset_category_breakdown",
        "lease_type",
    ],
    ComplianceFramework.ISO_14064: [
        "total_co2e",
        "uncertainty_analysis",
        "base_year",
        "methodology",
        "data_sources",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "total_co2e",
        "category_breakdown",
        "methodology",
        "targets",
        "actions",
        "asset_portfolio_summary",
    ],
    ComplianceFramework.CDP: [
        "total_co2e",
        "asset_category_breakdown",
        "lease_count",
        "data_coverage",
        "verification_status",
    ],
    ComplianceFramework.SBTI: [
        "total_co2e",
        "target_coverage",
        "reduction_initiatives",
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
        "intensity_ratios",
    ],
}


# ==============================================================================
# INPUT MODELS (12 total)
# ==============================================================================


class BuildingAssetInput(GreenLangBase):
    """
    Input for building/real-estate leased asset emissions calculation.

    Supports both asset-specific (metered data) and average-data (EUI benchmark)
    methods. For asset-specific, provide electricity_kwh and/or gas_kwh overrides.
    For average-data, floor_area_sqm and building_type/climate_zone are required.

    Example:
        >>> building = BuildingAssetInput(
        ...     asset_id="BLD-001",
        ...     building_type=BuildingType.OFFICE,
        ...     floor_area_sqm=Decimal("2500.0"),
        ...     climate_zone=ClimateZone.TEMPERATE,
        ...     country_code="GB",
        ...     lease_share=Decimal("1.0"),
        ...     method=CalculationMethod.AVERAGE_DATA,
        ... )
    """

    asset_id: str = Field(
        ..., min_length=1,
        description="Unique identifier for the leased building asset"
    )
    asset_name: Optional[str] = Field(
        default=None,
        description="Human-readable name for the leased building"
    )
    building_type: BuildingType = Field(
        ..., description="Type of building (office, retail, warehouse, etc.)"
    )
    floor_area_sqm: Decimal = Field(
        ..., gt=0,
        description="Total leased floor area in square metres"
    )
    climate_zone: Optional[ClimateZone] = Field(
        default=None,
        description="Climate zone (auto-resolved from country_code if not provided)"
    )
    country_code: str = Field(
        default="GLOBAL",
        description="ISO 3166-1 alpha-2 country code for grid EF and climate zone"
    )
    egrid_subregion: Optional[str] = Field(
        default=None,
        description="US eGRID subregion for finer-grained US grid EF"
    )
    lease_type: LeaseType = Field(
        default=LeaseType.OPERATING,
        description="Lease classification (operating or finance)"
    )
    lease_share: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Fraction of building leased by reporting company (0-1)"
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.AVERAGE_DATA,
        description="Calculation method to use"
    )
    electricity_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Override: metered annual electricity consumption (kWh)"
    )
    gas_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Override: metered annual natural gas consumption (kWh)"
    )
    other_fuel_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Override: other fuel consumption (kWh)"
    )
    other_fuel_type: Optional[EnergySource] = Field(
        default=None,
        description="Type of other fuel (heating_oil, lpg, coal, etc.)"
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.FLOOR_AREA,
        description="Allocation method for multi-tenant buildings"
    )
    allocation_factor: Optional[Decimal] = Field(
        default=None, ge=0, le=1,
        description="Custom allocation factor (required if allocation_method=custom)"
    )
    reporting_period: str = Field(
        default="2024",
        description="Reporting period (e.g., '2024', '2024-Q3')"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("floor_area_sqm")
    def validate_floor_area(cls, v: Decimal) -> Decimal:
        """Validate floor area is positive and reasonable."""
        if v <= 0:
            raise ValueError(f"Floor area must be positive, got {v}")
        if v > Decimal("1000000"):
            raise ValueError(
                f"Floor area exceeds 1,000,000 sqm: {v}. "
                "Please verify this is correct."
            )
        return v

    @validator("lease_share")
    def validate_lease_share(cls, v: Decimal) -> Decimal:
        """Validate lease share is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError(f"Lease share must be 0-1, got {v}")
        return v

    @validator("allocation_factor")
    def validate_allocation_factor(cls, v: Optional[Decimal], values: dict) -> Optional[Decimal]:
        """Validate allocation factor is provided when method is custom."""
        method = values.get("allocation_method")
        if method == AllocationMethod.CUSTOM and v is None:
            raise ValueError(
                "allocation_factor is required when allocation_method is 'custom'"
            )
        return v


class VehicleAssetInput(GreenLangBase):
    """
    Input for leased vehicle fleet emissions calculation.

    Supports distance-based (km driven) and fuel-based calculation methods.

    Example:
        >>> vehicle = VehicleAssetInput(
        ...     asset_id="VEH-001",
        ...     vehicle_type=VehicleType.MEDIUM_CAR,
        ...     fuel_type=FuelType.DIESEL,
        ...     annual_km=Decimal("25000.0"),
        ...     fleet_size=10,
        ...     method=CalculationMethod.ASSET_SPECIFIC,
        ... )
    """

    asset_id: str = Field(
        ..., min_length=1,
        description="Unique identifier for the leased vehicle or fleet"
    )
    asset_name: Optional[str] = Field(
        default=None,
        description="Human-readable name / description"
    )
    vehicle_type: VehicleType = Field(
        ..., description="Type of leased vehicle"
    )
    fuel_type: FuelType = Field(
        ..., description="Fuel type of the vehicle"
    )
    annual_km: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Annual distance driven per vehicle (km)"
    )
    annual_fuel_litres: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Annual fuel consumption per vehicle (litres)"
    )
    fleet_size: int = Field(
        default=1, ge=1,
        description="Number of vehicles of this type in the leased fleet"
    )
    lease_type: LeaseType = Field(
        default=LeaseType.OPERATING,
        description="Lease classification"
    )
    country_code: str = Field(
        default="GLOBAL",
        description="Country code for grid EF (BEV charging)"
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.ASSET_SPECIFIC,
        description="Calculation method"
    )
    reporting_period: str = Field(
        default="2024",
        description="Reporting period"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("annual_km")
    def validate_annual_km(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate annual km is reasonable."""
        if v is not None and v > Decimal("500000"):
            raise ValueError(
                f"Annual km exceeds 500,000: {v}. Please verify."
            )
        return v

    @validator("annual_fuel_litres")
    def validate_fuel(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate fuel consumption is reasonable."""
        if v is not None and v > Decimal("100000"):
            raise ValueError(
                f"Annual fuel exceeds 100,000 litres: {v}. Please verify."
            )
        return v


class EquipmentAssetInput(GreenLangBase):
    """
    Input for leased equipment emissions calculation.

    Supports both asset-specific (metered hours and fuel) and average-data
    (benchmark hours, load factor, fuel consumption rate) methods.

    Example:
        >>> equipment = EquipmentAssetInput(
        ...     asset_id="EQP-001",
        ...     equipment_type=EquipmentType.CONSTRUCTION,
        ...     operating_hours=Decimal("1200.0"),
        ...     fuel_consumption_litres=Decimal("22200.0"),
        ...     method=CalculationMethod.ASSET_SPECIFIC,
        ... )
    """

    asset_id: str = Field(
        ..., min_length=1,
        description="Unique identifier for the leased equipment"
    )
    asset_name: Optional[str] = Field(
        default=None,
        description="Human-readable name / description"
    )
    equipment_type: EquipmentType = Field(
        ..., description="Type of leased equipment"
    )
    quantity: int = Field(
        default=1, ge=1,
        description="Number of equipment units of this type"
    )
    operating_hours: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Annual operating hours per unit (override benchmark)"
    )
    load_factor: Optional[Decimal] = Field(
        default=None, gt=0, le=1,
        description="Load factor override (0-1)"
    )
    fuel_consumption_litres: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total annual fuel consumption (litres, for fuel-powered equipment)"
    )
    electricity_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total annual electricity consumption (kWh, for electric equipment)"
    )
    country_code: str = Field(
        default="GLOBAL",
        description="Country code for grid EF (electric equipment)"
    )
    lease_type: LeaseType = Field(
        default=LeaseType.OPERATING,
        description="Lease classification"
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.AVERAGE_DATA,
        description="Calculation method"
    )
    reporting_period: str = Field(
        default="2024",
        description="Reporting period"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("operating_hours")
    def validate_hours(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate operating hours is reasonable."""
        if v is not None and v > Decimal("8760"):
            raise ValueError(
                f"Operating hours exceeds 8,760 (hours in a year): {v}."
            )
        return v


class ITAssetInput(GreenLangBase):
    """
    Input for leased IT asset emissions calculation.

    Supports asset-specific (metered kWh) and average-data (power rating x
    utilization x PUE x hours) methods.

    Example:
        >>> it_asset = ITAssetInput(
        ...     asset_id="IT-001",
        ...     it_asset_type=ITAssetType.SERVER,
        ...     quantity=50,
        ...     country_code="US",
        ...     egrid_subregion="RFCW",
        ...     method=CalculationMethod.AVERAGE_DATA,
        ... )
    """

    asset_id: str = Field(
        ..., min_length=1,
        description="Unique identifier for the leased IT asset group"
    )
    asset_name: Optional[str] = Field(
        default=None,
        description="Human-readable name / description"
    )
    it_asset_type: ITAssetType = Field(
        ..., description="Type of leased IT asset"
    )
    quantity: int = Field(
        default=1, ge=1,
        description="Number of IT assets of this type"
    )
    power_kw_override: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Override typical power rating per unit (kW)"
    )
    utilization_override: Optional[Decimal] = Field(
        default=None, gt=0, le=1,
        description="Override utilization factor (0-1)"
    )
    pue_override: Optional[Decimal] = Field(
        default=None, ge=1,
        description="Override PUE (Power Usage Effectiveness, >= 1.0)"
    )
    annual_hours_override: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Override annual operating hours"
    )
    electricity_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Override: metered total annual electricity consumption (kWh)"
    )
    country_code: str = Field(
        default="GLOBAL",
        description="Country code for grid emission factor"
    )
    egrid_subregion: Optional[str] = Field(
        default=None,
        description="US eGRID subregion for finer-grained US grid EF"
    )
    lease_type: LeaseType = Field(
        default=LeaseType.OPERATING,
        description="Lease classification"
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.AVERAGE_DATA,
        description="Calculation method"
    )
    reporting_period: str = Field(
        default="2024",
        description="Reporting period"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("pue_override")
    def validate_pue(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate PUE is at least 1.0."""
        if v is not None and v < Decimal("1.0"):
            raise ValueError(
                f"PUE must be >= 1.0, got {v}"
            )
        return v

    @validator("annual_hours_override")
    def validate_hours(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate annual hours is reasonable."""
        if v is not None and v > Decimal("8760"):
            raise ValueError(
                f"Annual hours exceeds 8,760 (hours in a year): {v}."
            )
        return v


class LessorDataInput(GreenLangBase):
    """
    Input for lessor-specific emissions data (primary data from landlord).

    Used when the lessor provides total emissions or energy data directly.

    Example:
        >>> lessor = LessorDataInput(
        ...     asset_id="BLD-002",
        ...     lessor_name="Acme Properties Ltd",
        ...     total_co2e_kg=Decimal("85000.0"),
        ...     lease_share=Decimal("0.35"),
        ...     verification_status="third_party_verified",
        ... )
    """

    asset_id: str = Field(
        ..., min_length=1,
        description="Unique identifier for the leased asset"
    )
    lessor_name: Optional[str] = Field(
        default=None,
        description="Name of the lessor / landlord"
    )
    asset_category: AssetCategory = Field(
        default=AssetCategory.BUILDING,
        description="Asset category"
    )
    total_co2e_kg: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total annual CO2e provided by lessor (kgCO2e)"
    )
    total_electricity_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total annual electricity provided by lessor (kWh)"
    )
    total_gas_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total annual natural gas provided by lessor (kWh)"
    )
    lease_share: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Reporting company's share of the leased asset (0-1)"
    )
    country_code: str = Field(
        default="GLOBAL",
        description="Country code for grid EF if converting energy to emissions"
    )
    verification_status: Optional[str] = Field(
        default=None,
        description="Verification status (e.g., 'self_reported', 'third_party_verified')"
    )
    reporting_period: str = Field(
        default="2024",
        description="Reporting period"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("lease_share")
    def validate_lease_share(cls, v: Decimal) -> Decimal:
        """Validate lease share is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError(f"Lease share must be 0-1, got {v}")
        return v


class SpendInput(GreenLangBase):
    """
    Input for spend-based emissions calculation using EEIO factors.

    Used when only lease payment / rental spend data is available.

    Example:
        >>> spend = SpendInput(
        ...     asset_id="SPEND-001",
        ...     naics_code="531120",
        ...     amount=Decimal("500000.00"),
        ...     currency=CurrencyCode.USD,
        ...     reporting_year=2024,
        ... )
    """

    asset_id: str = Field(
        ..., min_length=1,
        description="Unique identifier for the spend record"
    )
    asset_category: AssetCategory = Field(
        default=AssetCategory.BUILDING,
        description="Asset category for the spend"
    )
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
    description: Optional[str] = Field(
        default=None,
        description="Description of the leased asset / service"
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
            raise ValueError(f"Spend amount must be positive, got {v}")
        return v

    @validator("naics_code")
    def validate_naics(cls, v: str) -> str:
        """Validate NAICS code exists in EEIO_FACTORS."""
        if v not in EEIO_FACTORS:
            raise ValueError(
                f"NAICS code '{v}' not found in EEIO_FACTORS. "
                f"Available codes: {sorted(EEIO_FACTORS.keys())}"
            )
        return v


class AssetInventoryInput(GreenLangBase):
    """
    Input for complete leased asset inventory (all categories combined).

    Wraps all asset types into a single submission for portfolio-level calculations.

    Example:
        >>> inventory = AssetInventoryInput(
        ...     inventory_id="INV-2024",
        ...     buildings=[building1, building2],
        ...     vehicles=[vehicle_fleet1],
        ...     reporting_period="2024",
        ... )
    """

    inventory_id: str = Field(
        ..., min_length=1,
        description="Unique identifier for the asset inventory"
    )
    buildings: List[BuildingAssetInput] = Field(
        default_factory=list,
        description="List of leased building assets"
    )
    vehicles: List[VehicleAssetInput] = Field(
        default_factory=list,
        description="List of leased vehicle assets"
    )
    equipment: List[EquipmentAssetInput] = Field(
        default_factory=list,
        description="List of leased equipment assets"
    )
    it_assets: List[ITAssetInput] = Field(
        default_factory=list,
        description="List of leased IT assets"
    )
    lessor_data: List[LessorDataInput] = Field(
        default_factory=list,
        description="List of lessor-provided emissions data"
    )
    spend_records: List[SpendInput] = Field(
        default_factory=list,
        description="List of spend-based records"
    )
    reporting_period: str = Field(
        default="2024",
        description="Reporting period (e.g., '2024', '2024-Q3')"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("buildings", "vehicles", "equipment", "it_assets",
               "lessor_data", "spend_records", pre=True, always=True)
    def validate_asset_list(cls, v: list) -> list:
        """Ensure asset list is a valid list (each list can be empty)."""
        if v is None:
            return []
        return v


class AllocationInput(GreenLangBase):
    """
    Input for emissions allocation across organizational units.

    Example:
        >>> allocation = AllocationInput(
        ...     total_co2e_kg=Decimal("150000.0"),
        ...     allocation_method=AllocationMethod.FLOOR_AREA,
        ...     shares={"BU-Sales": Decimal("0.40"), "BU-Engineering": Decimal("0.60")},
        ... )
    """

    total_co2e_kg: Decimal = Field(
        ..., ge=0,
        description="Total CO2e to allocate (kgCO2e)"
    )
    allocation_method: AllocationMethod = Field(
        ..., description="Allocation method"
    )
    shares: Dict[str, Decimal] = Field(
        ..., description="Allocation shares by entity (sum should equal 1.0)"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("shares")
    def validate_shares_sum(cls, v: Dict[str, Decimal]) -> Dict[str, Decimal]:
        """Validate allocation shares sum approximately to 1.0."""
        total = sum(v.values())
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Allocation shares must sum to ~1.0, got {total}"
            )
        return v


class BatchAssetInput(GreenLangBase):
    """
    Batch input for processing multiple individual assets in a single request.

    Example:
        >>> batch = BatchAssetInput(
        ...     batch_id="BATCH-2024-Q4",
        ...     buildings=[bld1, bld2],
        ...     vehicles=[veh1],
        ...     reporting_period="2024",
        ... )
    """

    batch_id: str = Field(
        ..., min_length=1,
        description="Unique batch identifier"
    )
    buildings: List[BuildingAssetInput] = Field(
        default_factory=list,
        description="Building assets to process"
    )
    vehicles: List[VehicleAssetInput] = Field(
        default_factory=list,
        description="Vehicle assets to process"
    )
    equipment: List[EquipmentAssetInput] = Field(
        default_factory=list,
        description="Equipment assets to process"
    )
    it_assets: List[ITAssetInput] = Field(
        default_factory=list,
        description="IT assets to process"
    )
    lessor_data: List[LessorDataInput] = Field(
        default_factory=list,
        description="Lessor-provided data records"
    )
    spend_records: List[SpendInput] = Field(
        default_factory=list,
        description="Spend-based records"
    )
    reporting_period: str = Field(
        ..., description="Reporting period"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceCheckInput(GreenLangBase):
    """
    Input for compliance checking against regulatory frameworks.

    Example:
        >>> check = ComplianceCheckInput(
        ...     frameworks=[
        ...         ComplianceFramework.GHG_PROTOCOL,
        ...         ComplianceFramework.CSRD_ESRS,
        ...     ],
        ...     total_co2e_kg=Decimal("250000.0"),
        ...     methods_used=["average_data", "asset_specific"],
        ...     ef_sources=["IEA 2024", "DEFRA 2024"],
        ... )
    """

    frameworks: List[ComplianceFramework] = Field(
        ..., min_length=1,
        description="Frameworks to check compliance against"
    )
    total_co2e_kg: Decimal = Field(
        ..., ge=0,
        description="Total reported CO2e (kgCO2e)"
    )
    methods_used: List[str] = Field(
        default_factory=list,
        description="Calculation methods used"
    )
    ef_sources: List[str] = Field(
        default_factory=list,
        description="Emission factor sources used"
    )
    asset_categories_covered: List[str] = Field(
        default_factory=list,
        description="Asset categories included in the calculation"
    )
    dqi_score: Optional[Decimal] = Field(
        default=None,
        description="Composite data quality indicator score (1-5)"
    )
    uncertainty_analysis: bool = Field(
        default=False,
        description="Whether uncertainty analysis was performed"
    )
    base_year: Optional[str] = Field(
        default=None,
        description="Base year for target tracking"
    )
    exclusions: List[str] = Field(
        default_factory=list,
        description="Any excluded asset categories with justification"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyInput(GreenLangBase):
    """
    Input for uncertainty quantification of emissions estimates.

    Example:
        >>> uncertainty = UncertaintyInput(
        ...     mean_co2e_kg=Decimal("250000.0"),
        ...     method=UncertaintyMethod.MONTE_CARLO,
        ...     calculation_method=CalculationMethod.AVERAGE_DATA,
        ...     data_quality_tier=DataQualityTier.CALCULATED,
        ...     iterations=10000,
        ... )
    """

    mean_co2e_kg: Decimal = Field(
        ..., ge=0,
        description="Mean emissions estimate (kgCO2e)"
    )
    method: UncertaintyMethod = Field(
        default=UncertaintyMethod.ANALYTICAL,
        description="Uncertainty quantification method"
    )
    calculation_method: CalculationMethod = Field(
        ..., description="Calculation method used (for uncertainty range lookup)"
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.CALCULATED,
        description="Data quality tier (for uncertainty range lookup)"
    )
    iterations: int = Field(
        default=10000, ge=100, le=1000000,
        description="Number of Monte Carlo iterations"
    )
    confidence_level: Decimal = Field(
        default=Decimal("0.95"), gt=0, lt=1,
        description="Confidence level for interval (e.g., 0.95)"
    )
    custom_half_width: Optional[Decimal] = Field(
        default=None, gt=0, lt=1,
        description="Custom uncertainty half-width override (fraction)"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class PortfolioInput(GreenLangBase):
    """
    Input for portfolio-level analysis across all leased assets.

    Provides organization-wide aggregation and hot-spot identification.

    Example:
        >>> portfolio = PortfolioInput(
        ...     portfolio_id="PORTFOLIO-2024",
        ...     organization_name="Acme Corp",
        ...     total_employees=5000,
        ...     total_revenue_usd=Decimal("500000000.0"),
        ...     inventories=[inventory1, inventory2],
        ...     reporting_period="2024",
        ... )
    """

    portfolio_id: str = Field(
        ..., min_length=1,
        description="Unique portfolio identifier"
    )
    organization_name: Optional[str] = Field(
        default=None,
        description="Organization name for reporting"
    )
    total_employees: Optional[int] = Field(
        default=None, gt=0,
        description="Total employees (for intensity ratio calculation)"
    )
    total_revenue_usd: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Total revenue in USD (for intensity ratio calculation)"
    )
    inventories: List[AssetInventoryInput] = Field(
        ..., min_length=1,
        description="Asset inventories to include in portfolio analysis"
    )
    compliance_frameworks: List[ComplianceFramework] = Field(
        default_factory=list,
        description="Frameworks to check compliance against"
    )
    reporting_period: str = Field(
        ..., description="Reporting period"
    )
    gwp_version: GWPVersion = Field(
        default=GWPVersion.AR6,
        description="GWP version to use for CO2e conversion"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


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
        >>> h = calculate_provenance_hash("BLD-001", Decimal("1234.56"))
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


def convert_currency_to_usd(
    amount: Decimal,
    currency: CurrencyCode,
    year: Optional[int] = None,
) -> Decimal:
    """
    Convert an amount from the given currency to USD using stored exchange rates.

    Optionally deflates to base year 2021 USD if a year is provided.

    Args:
        amount: Amount in the source currency.
        currency: Source currency code.
        year: Optional spend year for CPI deflation (base year 2021).

    Returns:
        Equivalent amount in USD, quantized to 8 decimal places.

    Raises:
        ValueError: If currency code is not found in CURRENCY_RATES.
        ValueError: If year is provided but not found in CPI_DEFLATORS.

    Example:
        >>> convert_currency_to_usd(Decimal("1000"), CurrencyCode.EUR)
        Decimal('1085.00000000')
        >>> convert_currency_to_usd(Decimal("1000"), CurrencyCode.USD, year=2024)
        Decimal('870.32201914')
    """
    rate = CURRENCY_RATES.get(currency)
    if rate is None:
        raise ValueError(
            f"Currency '{currency.value}' not found in CURRENCY_RATES"
        )
    usd_amount = amount * rate

    if year is not None:
        deflator = CPI_DEFLATORS.get(year)
        if deflator is None:
            raise ValueError(
                f"CPI deflator not available for year {year}. "
                f"Available years: {sorted(CPI_DEFLATORS.keys())}"
            )
        # Deflate to base year 2021 (deflator for 2021 = 1.0)
        usd_amount = usd_amount / deflator

    return usd_amount.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def resolve_climate_zone(
    country_code: str,
    climate_zone_override: Optional[ClimateZone] = None,
) -> str:
    """
    Resolve climate zone from country code or explicit override.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        climate_zone_override: Explicit climate zone (takes priority).

    Returns:
        Climate zone string value.

    Example:
        >>> resolve_climate_zone("US")
        'temperate'
        >>> resolve_climate_zone("US", ClimateZone.ARID)
        'arid'
    """
    if climate_zone_override is not None:
        return climate_zone_override.value
    return CLIMATE_ZONE_MAP.get(country_code, "temperate")


def get_grid_ef(
    country_code: str,
    egrid_subregion: Optional[str] = None,
) -> Decimal:
    """
    Get grid emission factor for a country or eGRID subregion.

    If a US eGRID subregion is provided, uses the subregional factor.
    Otherwise falls back to country-level or GLOBAL default.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        egrid_subregion: Optional US eGRID subregion acronym.

    Returns:
        Grid emission factor in kgCO2e per kWh.

    Example:
        >>> get_grid_ef("US")
        Decimal('0.37170000')
        >>> get_grid_ef("US", "CAMX")
        Decimal('0.22800000')
        >>> get_grid_ef("GLOBAL")
        Decimal('0.43600000')
    """
    if egrid_subregion is not None and egrid_subregion in US_EGRID_FACTORS:
        return US_EGRID_FACTORS[egrid_subregion]

    country_data = GRID_EMISSION_FACTORS.get(country_code)
    if country_data is not None:
        return country_data["ef_per_kwh"]

    return GRID_EMISSION_FACTORS["GLOBAL"]["ef_per_kwh"]


def get_grid_wtt_ef(
    country_code: str,
) -> Decimal:
    """
    Get well-to-tank (WTT) grid emission factor for a country.

    Falls back to GLOBAL default if country-specific data is unavailable.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        WTT grid emission factor in kgCO2e per kWh.

    Example:
        >>> get_grid_wtt_ef("US")
        Decimal('0.04851000')
    """
    country_data = GRID_EMISSION_FACTORS.get(country_code)
    if country_data is not None:
        return country_data["wtt_per_kwh"]
    return GRID_EMISSION_FACTORS["GLOBAL"]["wtt_per_kwh"]


def get_building_eui(
    building_type: str,
    climate_zone: str,
) -> Optional[Dict[str, Any]]:
    """
    Look up building EUI benchmark by building type and climate zone.

    Args:
        building_type: Building type value (e.g., "office").
        climate_zone: Climate zone value (e.g., "temperate").

    Returns:
        Dict with eui_kwh_sqm, gas_fraction, source; or None if not found.

    Example:
        >>> result = get_building_eui("office", "temperate")
        >>> result["eui_kwh_sqm"]
        Decimal('150.00000000')
    """
    return BUILDING_EUI_BENCHMARKS.get((building_type, climate_zone))


def get_vehicle_ef(
    vehicle_type: str,
    fuel_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Look up vehicle emission factor by vehicle type and fuel type.

    Args:
        vehicle_type: Vehicle type value (e.g., "medium_car").
        fuel_type: Fuel type value (e.g., "diesel").

    Returns:
        Dict with ef_per_km, wtt_per_km, source; or None if not found.

    Example:
        >>> result = get_vehicle_ef("medium_car", "diesel")
        >>> result["ef_per_km"]
        Decimal('0.16720000')
    """
    return VEHICLE_EMISSION_FACTORS.get((vehicle_type, fuel_type))


def get_equipment_benchmark(
    equipment_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Look up equipment benchmark by equipment type.

    Args:
        equipment_type: Equipment type value (e.g., "construction").

    Returns:
        Dict with default_hours, load_factor, fuel_type,
        fuel_consumption_l_per_hr, source; or None if not found.

    Example:
        >>> result = get_equipment_benchmark("construction")
        >>> result["fuel_consumption_l_per_hr"]
        Decimal('18.50000000')
    """
    return EQUIPMENT_BENCHMARKS.get(equipment_type)


def get_it_power_rating(
    it_asset_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Look up IT asset power rating by type.

    Args:
        it_asset_type: IT asset type value (e.g., "server").

    Returns:
        Dict with typical_power_kw, utilization, pue_default,
        annual_hours, source; or None if not found.

    Example:
        >>> result = get_it_power_rating("server")
        >>> result["typical_power_kw"]
        Decimal('0.50000000')
    """
    return IT_POWER_RATINGS.get(it_asset_type)
