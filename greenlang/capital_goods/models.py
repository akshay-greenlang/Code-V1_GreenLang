# -*- coding: utf-8 -*-
"""
Capital Goods Agent Data Models - AGENT-MRV-015

Pydantic v2 data models for the Capital Goods Agent SDK covering
GHG Protocol Scope 3 Category 2 emissions from upstream production
of capital goods (fixed assets / PP&E) purchased or acquired during
the reporting period.  Key design rules:

- 100 % of cradle-to-gate emissions reported in YEAR OF ACQUISITION
  (NO depreciation of emissions over useful life).
- Capital goods = PP&E in financial statements (buildings, machinery,
  equipment, vehicles, IT infrastructure, furniture & fixtures,
  land improvements, leasehold improvements).
- Double-counting prevention against Category 1 (purchased goods &
  services) AND Scope 1 / Scope 2 use-phase emissions.
- Four calculation methods identical to Category 1: spend-based
  (EEIO), average-data (physical EFs), supplier-specific
  (EPD/PCF/CDP), and hybrid (multi-method).

Enumerations (20):
    CalculationMethod, AssetCategory, AssetSubCategory,
    SpendClassificationSystem, EEIODatabase, PhysicalEFSource,
    SupplierDataSource, AllocationMethod, CurrencyCode,
    DQIDimension, DQIScore, UncertaintyMethod,
    ComplianceFramework, ComplianceStatus, PipelineStage,
    ExportFormat, BatchStatus, GWPSource, EmissionGas,
    CapitalizationPolicy

Constants (13):
    GWP_VALUES, DQI_SCORE_VALUES, DQI_QUALITY_TIERS,
    UNCERTAINTY_RANGES, COVERAGE_THRESHOLDS,
    EF_HIERARCHY_PRIORITY, PEDIGREE_UNCERTAINTY_FACTORS,
    CURRENCY_EXCHANGE_RATES, CAPITAL_SECTOR_MARGIN_PERCENTAGES,
    CAPITAL_EEIO_EMISSION_FACTORS, CAPITAL_PHYSICAL_EMISSION_FACTORS,
    ASSET_USEFUL_LIFE_RANGES, FRAMEWORK_REQUIRED_DISCLOSURES

Data Models (25):
    CapitalAssetRecord, CapExSpendRecord, PhysicalRecord,
    SupplierRecord, SpendBasedResult, AverageDataResult,
    SupplierSpecificResult, HybridResult, EEIOFactor,
    PhysicalEF, SupplierEF, DQIAssessment,
    AssetClassification, CapitalizationThreshold,
    UsefulLifeRange, DepreciationContext, MaterialityItem,
    CoverageReport, ComplianceRequirement, ComplianceCheckResult,
    CalculationRequest, BatchRequest, CalculationResult,
    AggregationResult, HotSpotAnalysis

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-S3-002)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Module-level Constants
# ---------------------------------------------------------------------------

#: Agent identifier for registry integration.
AGENT_ID: str = "GL-MRV-S3-002"

#: Agent component identifier.
AGENT_COMPONENT: str = "AGENT-MRV-015"

#: Service version string.
VERSION: str = "1.0.0"

#: Database table prefix for all Capital Goods tables.
TABLE_PREFIX: str = "gl_cg_"

#: Maximum number of capital asset records per calculation request.
MAX_ASSET_RECORDS: int = 100_000

#: Maximum number of periods in a batch request.
MAX_BATCH_PERIODS: int = 120

#: Maximum number of facilities per aggregation.
MAX_FACILITIES: int = 50_000

#: Maximum number of suppliers per request.
MAX_SUPPLIERS: int = 10_000

#: Maximum number of frameworks per compliance check.
MAX_FRAMEWORKS: int = 20

#: Maximum number of compliance requirements per framework.
MAX_REQUIREMENTS_PER_FRAMEWORK: int = 200

#: Maximum number of hot-spot items per analysis.
MAX_HOTSPOT_ITEMS: int = 5_000

#: Default confidence level for uncertainty quantification.
DEFAULT_CONFIDENCE_LEVEL: Decimal = Decimal("95.0")

#: Positive infinity sentinel for Decimal comparisons.
DECIMAL_INF: Decimal = Decimal("Infinity")

#: Number of decimal places for Decimal quantization.
DECIMAL_PLACES: int = 8

#: Decimal zero constant for arithmetic operations.
ZERO: Decimal = Decimal("0")

#: Decimal one constant.
ONE: Decimal = Decimal("1")

#: Decimal one hundred constant for percentage calculations.
ONE_HUNDRED: Decimal = Decimal("100")

#: Decimal one thousand constant for unit conversions.
ONE_THOUSAND: Decimal = Decimal("1000")


# =============================================================================
# Enumerations (20)
# =============================================================================


class CalculationMethod(str, Enum):
    """GHG Protocol Scope 3 Category 2 calculation methods.

    The GHG Protocol Technical Guidance defines four methods for
    Category 2 capital goods emissions, listed from most to least
    accurate.  Organizations should use the most accurate method
    for which data is available.  The same four methods apply as
    Category 1 but with capital-goods-specific EFs and sectors.

    SUPPLIER_SPECIFIC: Uses primary data from capital goods
        suppliers on cradle-to-gate emissions.  Highest accuracy
        (+/- 10-30 %).  Requires EPDs, PCFs, or CDP data.
    HYBRID: Combines all three methods using the best available
        data for each asset.  High accuracy (+/- 20-50 %).
    AVERAGE_DATA: Uses physical quantities (mass, area, units)
        multiplied by industry-average cradle-to-gate emission
        factors from LCA databases.  Medium accuracy (+/- 30-60 %).
    SPEND_BASED: Estimates emissions by multiplying CapEx in each
        sector by EEIO emission factors.  Lowest accuracy
        (+/- 50-100 %) but broadest coverage.
    """

    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"


class AssetCategory(str, Enum):
    """Top-level capital asset categories aligned with PP&E classes.

    Maps to standard PP&E line items in financial statements
    (IFRS IAS 16, US GAAP ASC 360).  Each category has distinct
    emission factor profiles, useful life ranges, and NAICS
    sector mappings for EEIO calculations.

    BUILDINGS: Owned buildings and structures (office, warehouse,
        manufacturing, retail).
    MACHINERY: Production machinery and heavy equipment (CNC,
        presses, cranes, conveyors).
    EQUIPMENT: Non-production equipment (HVAC, electrical panels,
        generators, compressors).
    VEHICLES: Owned fleet vehicles (passenger cars, trucks,
        forklifts, vans).
    IT_INFRASTRUCTURE: Servers, storage, networking, UPS, and
        other data-centre equipment.
    FURNITURE_FIXTURES: Office furniture, shelving, partitions,
        and permanent fixtures.
    LAND_IMPROVEMENTS: Site improvements (paving, landscaping,
        fencing, drainage).
    LEASEHOLD_IMPROVEMENTS: Improvements to leased premises
        (fit-out, partitions, utilities).
    """

    BUILDINGS = "buildings"
    MACHINERY = "machinery"
    EQUIPMENT = "equipment"
    VEHICLES = "vehicles"
    IT_INFRASTRUCTURE = "it_infrastructure"
    FURNITURE_FIXTURES = "furniture_fixtures"
    LAND_IMPROVEMENTS = "land_improvements"
    LEASEHOLD_IMPROVEMENTS = "leasehold_improvements"


class AssetSubCategory(str, Enum):
    """Detailed sub-categories for capital assets (~40 entries).

    Provides granular classification within each AssetCategory
    for more precise emission factor matching and hot-spot
    analysis.  Sub-categories align with common fixed-asset
    register classifications used in ERP systems.
    """

    # Buildings (4)
    OFFICE_BUILDING = "office_building"
    WAREHOUSE = "warehouse"
    MANUFACTURING_FACILITY = "manufacturing_facility"
    RETAIL_STORE = "retail_store"

    # Machinery (5)
    CNC_MACHINE = "cnc_machine"
    PRESS = "press"
    CRANE = "crane"
    CONVEYOR = "conveyor"
    INDUSTRIAL_ROBOT = "industrial_robot"

    # Equipment (5)
    HVAC = "hvac"
    ELECTRICAL_PANEL = "electrical_panel"
    GENERATOR = "generator"
    COMPRESSOR = "compressor"
    TRANSFORMER = "transformer"

    # Vehicles (5)
    PASSENGER_CAR = "passenger_car"
    LIGHT_TRUCK = "light_truck"
    HEAVY_TRUCK = "heavy_truck"
    FORKLIFT = "forklift"
    VAN = "van"

    # IT Infrastructure (5)
    SERVER = "server"
    NETWORK_SWITCH = "network_switch"
    STORAGE_ARRAY = "storage_array"
    UPS = "ups"
    RACK_ENCLOSURE = "rack_enclosure"

    # Furniture & Fixtures (4)
    OFFICE_DESK = "office_desk"
    OFFICE_CHAIR = "office_chair"
    SHELVING = "shelving"
    PARTITION = "partition"

    # Land Improvements (4)
    PAVING = "paving"
    LANDSCAPING = "landscaping"
    FENCING = "fencing"
    DRAINAGE = "drainage"

    # Leasehold Improvements (4)
    FITOUT_GENERAL = "fitout_general"
    INTERIOR_PARTITION = "interior_partition"
    FLOORING = "flooring"
    CEILING = "ceiling"

    # Cross-category specials (4)
    SOLAR_PANEL = "solar_panel"
    WIND_TURBINE = "wind_turbine"
    BATTERY_STORAGE = "battery_storage"
    ELECTRIC_MOTOR = "electric_motor"


class SpendClassificationSystem(str, Enum):
    """Industry classification systems for CapEx categorization.

    NAICS: North American Industry Classification System (2022).
    NACE: Statistical Classification of Economic Activities (EU).
    ISIC: International Standard Industrial Classification (UN).
    UNSPSC: UN Standard Products and Services Code v28.
    """

    NAICS = "naics"
    NACE = "nace"
    ISIC = "isic"
    UNSPSC = "unspsc"


class EEIODatabase(str, Enum):
    """Environmentally-Extended Input-Output databases for spend-based EFs.

    EPA_USEEIO: US EPA EEIO Model v1.2/v1.3 -- 1,016 commodities.
    EXIOBASE: EXIOBASE 3.8 multi-regional IO -- 163 x 49 regions.
    WIOD: World Input-Output Database (2016) -- 43 countries.
    GTAP: Global Trade Analysis Project v11 -- 141 regions.
    """

    EPA_USEEIO = "epa_useeio"
    EXIOBASE = "exiobase"
    WIOD = "wiod"
    GTAP = "gtap"


class PhysicalEFSource(str, Enum):
    """Sources of physical (quantity-based) emission factors.

    ECOINVENT: ecoinvent v3.11 LCA database.
    DEFRA: UK DEFRA/DESNZ emission conversion factors.
    ICE_DATABASE: Inventory of Carbon & Energy v3.0 (Univ. Bath).
    WORLD_STEEL: World Steel Association environmental data.
    IAI: International Aluminium Institute factors.
    CUSTOM: User-defined or custom emission factor source.
    """

    ECOINVENT = "ecoinvent"
    DEFRA = "defra"
    ICE_DATABASE = "ice_database"
    WORLD_STEEL = "world_steel"
    IAI = "iai"
    CUSTOM = "custom"


class SupplierDataSource(str, Enum):
    """Sources of supplier-specific emission data.

    EPD: Environmental Product Declarations (ISO 14025).
    PCF: Product Carbon Footprint per ISO 14067.
    CDP: CDP Supply Chain Program disclosure.
    ECOVADIS: EcoVadis sustainability ratings.
    DIRECT_MEASUREMENT: Direct supplier disclosure.
    ESTIMATED: Estimated from proxy data.
    """

    EPD = "epd"
    PCF = "pcf"
    CDP = "cdp"
    ECOVADIS = "ecovadis"
    DIRECT_MEASUREMENT = "direct_measurement"
    ESTIMATED = "estimated"


class AllocationMethod(str, Enum):
    """Allocation methods for supplier facility-level data.

    ECONOMIC: Allocate by economic value ratio.
    PHYSICAL: Allocate by physical output ratio.
    MASS: Allocate by mass ratio.
    ENERGY: Allocate by energy consumption ratio.
    HYBRID: Combined allocation using multiple bases.
    """

    ECONOMIC = "economic"
    PHYSICAL = "physical"
    MASS = "mass"
    ENERGY = "energy"
    HYBRID = "hybrid"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for CapEx spend-based calculations.

    Twenty major currencies supported.  All amounts are converted
    to USD (base currency) using annual average exchange rates
    before applying EEIO factors.
    """

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    KRW = "KRW"
    SGD = "SGD"
    HKD = "HKD"
    NZD = "NZD"
    MXN = "MXN"
    BRL = "BRL"
    INR = "INR"
    ZAR = "ZAR"
    THB = "THB"


class DQIDimension(str, Enum):
    """Data quality indicator dimensions per GHG Protocol Scope 3.

    Five dimensions scored on a 1-5 scale (lower is better).

    TEMPORAL: Timeliness relative to reporting year.
    GEOGRAPHICAL: Geographic representativeness of EF.
    TECHNOLOGICAL: Technology representativeness of EF.
    COMPLETENESS: Coverage of all relevant emission sources.
    RELIABILITY: Trustworthiness of data source and method.
    """

    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    TECHNOLOGICAL = "technological"
    COMPLETENESS = "completeness"
    RELIABILITY = "reliability"


class DQIScore(int, Enum):
    """Data quality score levels (1-5 scale, lower is better).

    VERY_GOOD: Score 1 -- verified primary data.
    GOOD: Score 2 -- established databases.
    FAIR: Score 3 -- industry averages.
    POOR: Score 4 -- broad estimates.
    VERY_POOR: Score 5 -- unverified assumptions.
    """

    VERY_GOOD = 1
    GOOD = 2
    FAIR = 3
    POOR = 4
    VERY_POOR = 5


class UncertaintyMethod(str, Enum):
    """Methods for quantifying uncertainty in emission calculations.

    MONTE_CARLO: Monte Carlo simulation (10,000+ iterations).
    ANALYTICAL: Analytical error propagation (root-sum-of-squares).
    TIER_DEFAULT: Default uncertainty ranges per calculation method.
    """

    MONTE_CARLO = "monte_carlo"
    ANALYTICAL = "analytical"
    TIER_DEFAULT = "tier_default"


class ComplianceFramework(str, Enum):
    """Regulatory and voluntary reporting frameworks for Category 2.

    GHG_PROTOCOL: GHG Protocol Scope 3 Standard Chapter 5.
    CSRD: EU CSRD ESRS E1 Scope 3 by category.
    CDP: Carbon Disclosure Project climate questionnaire.
    SBTI: Science Based Targets initiative.
    SB_253: California Senate Bill 253.
    GRI: Global Reporting Initiative Standard 305.
    ISO_14064: ISO 14064-1:2018 indirect emissions.
    """

    GHG_PROTOCOL = "ghg_protocol"
    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"
    ISO_14064 = "iso_14064"


class ComplianceStatus(str, Enum):
    """Result of a regulatory compliance check.

    COMPLIANT: All requirements met.
    PARTIAL: Some requirements met but gaps remain.
    NON_COMPLIANT: One or more mandatory requirements not met.
    """

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"


class PipelineStage(str, Enum):
    """Stages in the Capital Goods calculation pipeline.

    Ten sequential stages from validation through sealing.
    """

    VALIDATE = "validate"
    CLASSIFY_ASSETS = "classify_assets"
    RESOLVE_EFS = "resolve_efs"
    SPEND_CALC = "spend_calc"
    AVERAGE_CALC = "average_calc"
    SUPPLIER_CALC = "supplier_calc"
    HYBRID_AGGREGATE = "hybrid_aggregate"
    COMPLIANCE = "compliance"
    AGGREGATE = "aggregate"
    SEAL = "seal"


class ExportFormat(str, Enum):
    """Supported export formats for calculation outputs.

    JSON: Machine-readable JSON for API consumers.
    CSV: Comma-separated values for spreadsheet import.
    XLSX: Excel workbook with formatted sheets.
    PDF: PDF report with charts and tables.
    """

    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    PDF = "pdf"


class BatchStatus(str, Enum):
    """Status of a batch calculation job.

    PENDING: Created but not started.
    RUNNING: Actively processing.
    COMPLETED: All periods completed successfully.
    FAILED: All periods failed.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GWPSource(str, Enum):
    """IPCC Assessment Report source for Global Warming Potential.

    AR4: Fourth Assessment Report (2007).
    AR5: Fifth Assessment Report (2014).
    AR6: Sixth Assessment Report (2021) -- 100-year.
    AR6_20YR: Sixth Assessment Report -- 20-year GWP.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in Scope 3 Category 2 calculations.

    CO2: Carbon dioxide.
    CH4: Methane.
    N2O: Nitrous oxide.
    CO2E: Carbon dioxide equivalent (aggregate metric).
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO2E = "CO2e"


class CapitalizationPolicy(str, Enum):
    """Accounting capitalization policy governing asset recognition.

    Determines the threshold and rules for classifying an
    expenditure as a capital good (PP&E) vs. an operating
    expense (Category 1).

    COMPANY_DEFINED: Company-specific capitalization policy.
    IFRS: IFRS IAS 16 Property, Plant and Equipment.
    US_GAAP: US GAAP ASC 360 Property, Plant and Equipment.
    LOCAL_GAAP: Local GAAP capitalization rules.
    """

    COMPANY_DEFINED = "company_defined"
    IFRS = "ifrs"
    US_GAAP = "us_gaap"
    LOCAL_GAAP = "local_gaap"


# =============================================================================
# Constant Tables (13) -- all Decimal for deterministic arithmetic
# =============================================================================


# ---------------------------------------------------------------------------
# 1. GWP values by IPCC Assessment Report
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[GWPSource, Dict[EmissionGas, Decimal]] = {
    GWPSource.AR4: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("25"),
        EmissionGas.N2O: Decimal("298"),
        EmissionGas.CO2E: Decimal("1"),
    },
    GWPSource.AR5: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("28"),
        EmissionGas.N2O: Decimal("265"),
        EmissionGas.CO2E: Decimal("1"),
    },
    GWPSource.AR6: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("27.9"),
        EmissionGas.N2O: Decimal("273"),
        EmissionGas.CO2E: Decimal("1"),
    },
    GWPSource.AR6_20YR: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("82.5"),
        EmissionGas.N2O: Decimal("273"),
        EmissionGas.CO2E: Decimal("1"),
    },
}


# ---------------------------------------------------------------------------
# 2. DQI score numeric values (1=best, 5=worst)
# ---------------------------------------------------------------------------

DQI_SCORE_VALUES: Dict[DQIScore, Decimal] = {
    DQIScore.VERY_GOOD: Decimal("1.0"),
    DQIScore.GOOD: Decimal("2.0"),
    DQIScore.FAIR: Decimal("3.0"),
    DQIScore.POOR: Decimal("4.0"),
    DQIScore.VERY_POOR: Decimal("5.0"),
}


# ---------------------------------------------------------------------------
# 3. DQI quality tier labels with composite score ranges
#    (min_score inclusive, max_score exclusive)
# ---------------------------------------------------------------------------

DQI_QUALITY_TIERS: Dict[str, Tuple[Decimal, Decimal]] = {
    "Very Good": (Decimal("1.0"), Decimal("1.6")),
    "Good": (Decimal("1.6"), Decimal("2.6")),
    "Fair": (Decimal("2.6"), Decimal("3.6")),
    "Poor": (Decimal("3.6"), Decimal("4.6")),
    "Very Poor": (Decimal("4.6"), Decimal("5.1")),
}


# ---------------------------------------------------------------------------
# 4. Uncertainty ranges by calculation method (min%, max%)
# ---------------------------------------------------------------------------

UNCERTAINTY_RANGES: Dict[CalculationMethod, Tuple[Decimal, Decimal]] = {
    CalculationMethod.SUPPLIER_SPECIFIC: (Decimal("10"), Decimal("30")),
    CalculationMethod.HYBRID: (Decimal("20"), Decimal("50")),
    CalculationMethod.AVERAGE_DATA: (Decimal("30"), Decimal("60")),
    CalculationMethod.SPEND_BASED: (Decimal("50"), Decimal("100")),
}


# ---------------------------------------------------------------------------
# 5. Coverage thresholds by level (minimum CapEx percentage)
# ---------------------------------------------------------------------------

COVERAGE_THRESHOLDS: Dict[str, Decimal] = {
    "full": Decimal("100.0"),
    "high": Decimal("95.0"),
    "medium": Decimal("90.0"),
    "low": Decimal("80.0"),
    "minimal": Decimal("0.0"),
}


# ---------------------------------------------------------------------------
# 6. Emission factor hierarchy priority (1=best, 8=worst)
#    Per GHG Protocol Scope 3 Technical Guidance Section 1.4
# ---------------------------------------------------------------------------

EF_HIERARCHY_PRIORITY: Dict[str, int] = {
    "supplier_epd_verified": 1,
    "supplier_pcf_verified": 2,
    "supplier_cdp_unverified": 3,
    "product_lca_ecoinvent": 4,
    "material_avg_ice_defra": 5,
    "industry_avg_physical": 6,
    "regional_eeio_exiobase": 7,
    "global_avg_eeio_fallback": 8,
}


# ---------------------------------------------------------------------------
# 7. Pedigree matrix uncertainty factors by DQI score
#    Per ecoinvent pedigree matrix methodology
# ---------------------------------------------------------------------------

PEDIGREE_UNCERTAINTY_FACTORS: Dict[DQIScore, Decimal] = {
    DQIScore.VERY_GOOD: Decimal("1.00"),
    DQIScore.GOOD: Decimal("1.05"),
    DQIScore.FAIR: Decimal("1.10"),
    DQIScore.POOR: Decimal("1.20"),
    DQIScore.VERY_POOR: Decimal("1.50"),
}


# ---------------------------------------------------------------------------
# 8. Currency exchange rates to USD (annual average 2024 estimates)
#    All rates expressed as units of foreign currency per 1 USD
# ---------------------------------------------------------------------------

CURRENCY_EXCHANGE_RATES: Dict[CurrencyCode, Decimal] = {
    CurrencyCode.USD: Decimal("1.00000000"),
    CurrencyCode.EUR: Decimal("0.92410000"),
    CurrencyCode.GBP: Decimal("0.79250000"),
    CurrencyCode.JPY: Decimal("151.35000000"),
    CurrencyCode.CNY: Decimal("7.24500000"),
    CurrencyCode.CAD: Decimal("1.36200000"),
    CurrencyCode.AUD: Decimal("1.53500000"),
    CurrencyCode.CHF: Decimal("0.88150000"),
    CurrencyCode.SEK: Decimal("10.51200000"),
    CurrencyCode.NOK: Decimal("10.72800000"),
    CurrencyCode.DKK: Decimal("6.89400000"),
    CurrencyCode.KRW: Decimal("1345.60000000"),
    CurrencyCode.SGD: Decimal("1.34400000"),
    CurrencyCode.HKD: Decimal("7.82600000"),
    CurrencyCode.NZD: Decimal("1.63800000"),
    CurrencyCode.MXN: Decimal("17.14500000"),
    CurrencyCode.BRL: Decimal("4.97200000"),
    CurrencyCode.INR: Decimal("83.35000000"),
    CurrencyCode.ZAR: Decimal("18.65200000"),
    CurrencyCode.THB: Decimal("35.72000000"),
}


# ---------------------------------------------------------------------------
# 9. Capital-goods sector margin percentages
#    Used to convert purchaser price to producer/basic price for
#    capital-goods-specific EEIO factors.
# ---------------------------------------------------------------------------

CAPITAL_SECTOR_MARGIN_PERCENTAGES: Dict[str, Decimal] = {
    "construction": Decimal("15.0"),
    "machinery": Decimal("20.0"),
    "electronics": Decimal("25.0"),
    "vehicles": Decimal("18.0"),
    "it_equipment": Decimal("22.0"),
    "furniture": Decimal("30.0"),
    "hvac_systems": Decimal("17.0"),
    "electrical_equipment": Decimal("19.0"),
    "medical_instruments": Decimal("35.0"),
    "mining_equipment": Decimal("16.0"),
    "textile_machinery": Decimal("21.0"),
    "pumps_compressors": Decimal("18.0"),
    "turbines_generators": Decimal("14.0"),
    "aircraft": Decimal("12.0"),
    "railroad_equipment": Decimal("13.0"),
    "marine_vessels": Decimal("11.0"),
    "cement_materials": Decimal("10.0"),
    "steel_fabrication": Decimal("15.0"),
    "aluminum_products": Decimal("18.0"),
    "renewable_energy": Decimal("16.0"),
    "transformers": Decimal("17.0"),
    "cranes_hoists": Decimal("19.0"),
    "metalworking": Decimal("20.0"),
    "fluid_power": Decimal("22.0"),
    "general_industrial": Decimal("20.0"),
}


# ---------------------------------------------------------------------------
# 10. Capital-goods EEIO emission factors by NAICS-6 code
#     Source: EPA USEEIO v1.2, factors in kgCO2e per USD (purchaser
#     price, 2021 USD).  Capital-goods-specific sectors (50+ entries).
# ---------------------------------------------------------------------------

CAPITAL_EEIO_EMISSION_FACTORS: Dict[str, Decimal] = {
    # Construction (NAICS 236)
    "236210": Decimal("0.48"),    # Industrial Building Construction
    "236220": Decimal("0.42"),    # Commercial Building Construction
    "236116": Decimal("0.39"),    # New Multifamily Housing Construction
    "236118": Decimal("0.37"),    # Residential Remodelers
    "237110": Decimal("0.44"),    # Water & Sewer Line Construction
    "237310": Decimal("0.46"),    # Highway, Street, Bridge Construction
    # Machinery Manufacturing (NAICS 333)
    "333120": Decimal("0.35"),    # Construction Machinery Manufacturing
    "333131": Decimal("0.37"),    # Mining Machinery Manufacturing
    "333249": Decimal("0.32"),    # Other Industrial Machinery Mfg
    "333318": Decimal("0.31"),    # Other Textile Machinery Mfg (textile)
    "333413": Decimal("0.27"),    # Industrial Pumps Manufacturing
    "333415": Decimal("0.33"),    # HVAC & Commercial Refrigeration
    "333511": Decimal("0.30"),    # Industrial Mold Manufacturing
    "333515": Decimal("0.34"),    # Metalworking Machinery Manufacturing
    "333611": Decimal("0.39"),    # Turbine & Power Transmission Mfg
    "333923": Decimal("0.36"),    # Overhead Cranes & Hoists Mfg
    "333996": Decimal("0.26"),    # Fluid Power Equipment Mfg
    # Computer & Electronic (NAICS 334)
    "334111": Decimal("0.28"),    # Electronic Computer Manufacturing
    "334112": Decimal("0.26"),    # Computer Storage Device Mfg
    "334118": Decimal("0.25"),    # Computer Terminal & Printer Mfg
    "334210": Decimal("0.24"),    # Telephone Apparatus Manufacturing
    "334413": Decimal("0.22"),    # Semiconductor Device Manufacturing
    "334511": Decimal("0.20"),    # Search & Navigation Equipment Mfg
    # Electrical Equipment (NAICS 335)
    "335110": Decimal("0.28"),    # Electric Lamp Bulb & Parts Mfg
    "335311": Decimal("0.30"),    # Power & Distribution Transformers
    "335312": Decimal("0.29"),    # Motor & Generator Manufacturing
    "335999": Decimal("0.29"),    # Other Electrical Equipment Mfg
    # Transportation Equipment (NAICS 336)
    "336111": Decimal("0.38"),    # Automobile Manufacturing
    "336112": Decimal("0.36"),    # Light Truck & Utility Vehicle Mfg
    "336120": Decimal("0.41"),    # Heavy Duty Truck Manufacturing
    "336211": Decimal("0.33"),    # Motor Vehicle Body Manufacturing
    "336340": Decimal("0.31"),    # Motor Vehicle Brake System Mfg
    "336411": Decimal("0.45"),    # Aircraft Manufacturing
    "336510": Decimal("0.40"),    # Railroad Rolling Stock Mfg
    "336612": Decimal("0.43"),    # Boat Building
    # Furniture (NAICS 337)
    "337110": Decimal("0.18"),    # Wood Kitchen Cabinet & Counter Mfg
    "337121": Decimal("0.16"),    # Upholstered Household Furniture Mfg
    "337211": Decimal("0.15"),    # Wood Office Furniture Manufacturing
    "337214": Decimal("0.17"),    # Office Furniture (Non-Wood) Mfg
    # Medical Instruments (NAICS 339)
    "339112": Decimal("0.18"),    # Surgical & Medical Instrument Mfg
    "339113": Decimal("0.19"),    # Surgical Appliance & Supplies Mfg
    # Primary Metals (NAICS 327, 331)
    "327310": Decimal("0.52"),    # Cement Manufacturing
    "327320": Decimal("0.48"),    # Ready-Mix Concrete Manufacturing
    "331110": Decimal("0.55"),    # Iron & Steel Mills & Ferroalloy Mfg
    "331210": Decimal("0.42"),    # Iron & Steel Pipe & Tube Mfg
    "331313": Decimal("0.48"),    # Alumina Refining & Primary Aluminum
    "331420": Decimal("0.38"),    # Copper Rolling, Drawing, Extruding
    "331511": Decimal("0.35"),    # Iron Foundries
    # Miscellaneous Manufacturing (NAICS 332)
    "332111": Decimal("0.32"),    # Iron & Steel Forging
    "332312": Decimal("0.28"),    # Fabricated Structural Metal Mfg
    "332313": Decimal("0.27"),    # Plate Work Manufacturing
    "332410": Decimal("0.25"),    # Power Boiler & Heat Exchanger Mfg
    "332710": Decimal("0.23"),    # Machine Shops
    "332996": Decimal("0.26"),    # Fabricated Pipe & Pipe Fitting Mfg
}


# ---------------------------------------------------------------------------
# 11. Capital-goods physical emission factors
#     kgCO2e per kg (or per unit where noted) -- cradle-to-gate
#     Sources: ICE v3.0, World Steel 2023, IAI 2023, GCCA 2023,
#     ecoinvent 3.11, DEFRA 2023
# ---------------------------------------------------------------------------

CAPITAL_PHYSICAL_EMISSION_FACTORS: Dict[str, Decimal] = {
    # Structural metals
    "structural_steel": Decimal("1.55"),
    "reinforcing_steel": Decimal("1.99"),
    "stainless_steel": Decimal("6.15"),
    "aluminum_sheet": Decimal("8.24"),
    "aluminum_extrusion": Decimal("6.67"),
    "copper_pipe": Decimal("3.81"),
    "copper_wire": Decimal("3.64"),
    # Concrete & masite
    "concrete_25mpa": Decimal("0.132"),
    "concrete_32mpa": Decimal("0.163"),
    "concrete_40mpa": Decimal("0.188"),
    "concrete_precast": Decimal("0.176"),
    "brick": Decimal("0.24"),
    # Glass
    "glass_float": Decimal("1.22"),
    "glass_tempered": Decimal("1.67"),
    "glass_double_glazed": Decimal("2.89"),
    # Timber
    "timber_softwood": Decimal("0.51"),
    "timber_hardwood": Decimal("0.86"),
    "timber_glulam": Decimal("0.45"),
    # Interior materials
    "plasterboard": Decimal("0.39"),
    "ceramic_tiles": Decimal("0.74"),
    "carpet": Decimal("5.88"),
    "vinyl_flooring": Decimal("4.21"),
    "paint_water_based": Decimal("2.41"),
    "paint_solvent_based": Decimal("3.56"),
    # Insulation
    "insulation_mineral_wool": Decimal("1.28"),
    "insulation_eps": Decimal("3.29"),
    "insulation_xps": Decimal("3.48"),
    "insulation_pir": Decimal("3.44"),
    # Piping & roofing
    "pvc_pipe": Decimal("3.23"),
    "hdpe_pipe": Decimal("2.52"),
    "roofing_membrane": Decimal("4.12"),
    "asphalt": Decimal("0.043"),
    # IT & electronics (per unit)
    "led_panel_per_unit": Decimal("25.0"),
    "server_per_unit": Decimal("500.0"),
    "laptop_per_unit": Decimal("350.0"),
    "desktop_per_unit": Decimal("280.0"),
    "monitor_per_unit": Decimal("180.0"),
    "network_switch_per_unit": Decimal("120.0"),
    "ups_per_unit": Decimal("450.0"),
    # Renewable energy & storage (per kW / MW / kWh)
    "solar_panel_per_kw": Decimal("1200.0"),
    "wind_turbine_per_mw": Decimal("350000.0"),
    "battery_li_ion_per_kwh": Decimal("75.0"),
    # Electrical (per unit / kW)
    "transformer_per_unit": Decimal("800.0"),
    "electric_motor_per_kw": Decimal("15.0"),
}


# ---------------------------------------------------------------------------
# 12. Asset useful life ranges by category
#     (min_years, max_years, default_years)
# ---------------------------------------------------------------------------

ASSET_USEFUL_LIFE_RANGES: Dict[str, Tuple[int, int, int]] = {
    AssetCategory.BUILDINGS.value: (20, 50, 40),
    AssetCategory.MACHINERY.value: (7, 20, 15),
    AssetCategory.EQUIPMENT.value: (5, 15, 10),
    AssetCategory.VEHICLES.value: (3, 10, 7),
    AssetCategory.IT_INFRASTRUCTURE.value: (3, 7, 5),
    AssetCategory.FURNITURE_FIXTURES.value: (5, 15, 10),
    AssetCategory.LAND_IMPROVEMENTS.value: (10, 30, 20),
    AssetCategory.LEASEHOLD_IMPROVEMENTS.value: (5, 15, 10),
    # Sub-category overrides
    AssetSubCategory.OFFICE_BUILDING.value: (25, 50, 40),
    AssetSubCategory.WAREHOUSE.value: (20, 40, 30),
    AssetSubCategory.MANUFACTURING_FACILITY.value: (20, 50, 35),
    AssetSubCategory.RETAIL_STORE.value: (15, 30, 25),
    AssetSubCategory.CNC_MACHINE.value: (8, 20, 15),
    AssetSubCategory.PRESS.value: (10, 25, 18),
    AssetSubCategory.CRANE.value: (10, 25, 20),
    AssetSubCategory.CONVEYOR.value: (8, 18, 12),
    AssetSubCategory.INDUSTRIAL_ROBOT.value: (5, 15, 10),
    AssetSubCategory.HVAC.value: (10, 20, 15),
    AssetSubCategory.GENERATOR.value: (10, 25, 18),
    AssetSubCategory.COMPRESSOR.value: (8, 20, 12),
    AssetSubCategory.TRANSFORMER.value: (15, 30, 25),
    AssetSubCategory.PASSENGER_CAR.value: (3, 8, 5),
    AssetSubCategory.LIGHT_TRUCK.value: (4, 10, 7),
    AssetSubCategory.HEAVY_TRUCK.value: (5, 12, 8),
    AssetSubCategory.FORKLIFT.value: (5, 12, 8),
    AssetSubCategory.SERVER.value: (3, 7, 5),
    AssetSubCategory.NETWORK_SWITCH.value: (3, 7, 5),
    AssetSubCategory.STORAGE_ARRAY.value: (3, 7, 5),
    AssetSubCategory.UPS.value: (5, 10, 7),
    AssetSubCategory.SOLAR_PANEL.value: (20, 30, 25),
    AssetSubCategory.WIND_TURBINE.value: (20, 30, 25),
    AssetSubCategory.BATTERY_STORAGE.value: (8, 15, 10),
    AssetSubCategory.ELECTRIC_MOTOR.value: (10, 20, 15),
}


# ---------------------------------------------------------------------------
# 13. Framework required disclosures for Category 2 compliance
# ---------------------------------------------------------------------------

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[
    ComplianceFramework, List[str]
] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "category_2_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "capex_coverage_percentage",
        "category_boundary_description",
        "double_counting_prevention_cat1",
        "double_counting_prevention_scope1_2",
        "base_year_recalculation_policy",
        "exclusions_and_limitations",
        "gwp_values_used",
        "uncertainty_assessment",
        "year_of_acquisition_reporting",
    ],
    ComplianceFramework.CSRD: [
        "category_2_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "capex_coverage_percentage",
        "supplier_engagement_strategy",
        "intensity_per_revenue",
        "significant_changes_explanation",
        "base_year_emissions",
        "reduction_targets",
        "value_chain_boundary",
        "gwp_values_used",
        "capital_goods_classification",
    ],
    ComplianceFramework.CDP: [
        "category_2_total_tco2e",
        "category_2_relevance",
        "calculation_methodology",
        "percentage_calculated_primary_data",
        "percentage_calculated_secondary_data",
        "emission_factor_sources",
        "capex_coverage_percentage",
        "verification_status",
        "boundary_description",
        "year_of_acquisition_reporting",
    ],
    ComplianceFramework.SBTI: [
        "category_2_total_tco2e",
        "base_year_category_2_tco2e",
        "target_year",
        "reduction_percentage",
        "coverage_percentage",
        "supplier_engagement_targets",
        "calculation_methodology",
        "emission_factor_sources",
    ],
    ComplianceFramework.SB_253: [
        "category_2_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "capex_coverage_percentage",
        "assurance_provider",
        "reporting_entity_revenue",
        "gwp_values_used",
    ],
    ComplianceFramework.GRI: [
        "category_2_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "gwp_values_used",
        "consolidation_approach",
        "base_year_information",
        "standards_and_methodologies",
        "significant_changes",
    ],
    ComplianceFramework.ISO_14064: [
        "category_2_total_tco2e",
        "method_justification",
        "emission_by_gas_co2",
        "emission_by_gas_ch4",
        "emission_by_gas_n2o",
        "emission_factor_sources",
        "gwp_values_used",
        "uncertainty_assessment",
        "organizational_boundary",
        "reporting_period",
        "base_year_information",
        "data_quality_assessment",
    ],
}


# =============================================================================
# Data Models (25) -- Pydantic v2, frozen=True
# =============================================================================


# ---------------------------------------------------------------------------
# 1. CapitalAssetRecord
# ---------------------------------------------------------------------------


class CapitalAssetRecord(BaseModel):
    """A single capital asset (PP&E) record for emission calculation.

    Represents one fixed asset from the asset register.  Contains
    acquisition cost, physical attributes, classification codes,
    and capitalization policy information.  This is the primary
    input unit for the Capital Goods calculation pipeline.

    CRITICAL: 100 % of cradle-to-gate emissions are reported in
    the year of acquisition.  There is NO depreciation of
    emissions over the useful life.

    Attributes:
        asset_id: Unique identifier from fixed-asset register.
        asset_category: Top-level PP&E category.
        subcategory: Detailed sub-category for EF matching.
        description: Human-readable asset description.
        acquisition_date: Date the asset was acquired / placed
            in service.
        capex_amount: Capital expenditure in original currency.
        currency: Currency of the CapEx amount.
        quantity: Number of units acquired (default 1).
        unit: Unit of measurement (e.g. unit, kg, m2, kW).
        weight_kg: Total weight in kilograms (optional).
        useful_life_years: Expected useful life in years.
        capitalization_policy: Accounting policy for capitalization.
        naics_code: NAICS 2022 code (6-digit) for EEIO matching.
        nace_code: NACE Rev 2.1 code for EXIOBASE matching.
        supplier_id: Unique identifier of the asset supplier.
        supplier_name: Human-readable supplier name.
        facility_id: Facility where asset is installed.
        is_leased: Whether asset is a finance lease.
        is_second_hand: Whether asset is used / refurbished.
        is_intercompany: Whether this is an intercompany transfer.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    asset_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier from fixed-asset register",
    )
    asset_category: AssetCategory = Field(
        ...,
        description="Top-level PP&E category",
    )
    subcategory: Optional[AssetSubCategory] = Field(
        default=None,
        description="Detailed sub-category for EF matching",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Human-readable asset description",
    )
    acquisition_date: date = Field(
        ...,
        description="Date asset was acquired / placed in service",
    )
    capex_amount: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Capital expenditure in original currency",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of the CapEx amount",
    )
    quantity: Decimal = Field(
        default=Decimal("1"),
        gt=Decimal("0"),
        description="Number of units acquired",
    )
    unit: str = Field(
        default="unit",
        max_length=50,
        description="Unit of measurement (e.g. unit, kg, m2, kW)",
    )
    weight_kg: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Total weight in kilograms (optional)",
    )
    useful_life_years: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Expected useful life in years",
    )
    capitalization_policy: CapitalizationPolicy = Field(
        default=CapitalizationPolicy.COMPANY_DEFINED,
        description="Accounting policy for capitalization",
    )
    naics_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="NAICS 2022 code (6-digit) for EEIO matching",
    )
    nace_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="NACE Rev 2.1 code for EXIOBASE matching",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Unique identifier of the asset supplier",
    )
    supplier_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable supplier name",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Facility where asset is installed",
    )
    is_leased: bool = Field(
        default=False,
        description="Whether asset is a finance lease",
    )
    is_second_hand: bool = Field(
        default=False,
        description="Whether asset is used / refurbished",
    )
    is_intercompany: bool = Field(
        default=False,
        description="Whether this is an intercompany transfer",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for extensibility",
    )

    @field_validator("acquisition_date")
    @classmethod
    def _validate_acquisition_date(cls, v: date) -> date:
        """Validate acquisition date is not in the far future."""
        max_date = date(2100, 12, 31)
        if v > max_date:
            raise ValueError(
                f"acquisition_date ({v}) must not be after {max_date}"
            )
        return v


# ---------------------------------------------------------------------------
# 2. CapExSpendRecord
# ---------------------------------------------------------------------------


class CapExSpendRecord(BaseModel):
    """Spend-based input record for capital goods EEIO calculation.

    A capital asset record enriched with spend-specific fields
    needed for the spend-based calculation method: currency
    conversion parameters, EEIO database selection, and sector
    margin adjustment for capital goods.

    Attributes:
        record_id: Unique record identifier.
        asset_id: Reference to the capital asset record.
        amount: CapEx amount in original currency.
        currency: Currency of the CapEx amount.
        acquisition_year: Year of acquisition for reporting.
        spend_usd: CapEx converted to USD.
        spend_producer_usd: CapEx adjusted to producer price.
        naics_code: NAICS 6-digit sector code.
        nace_code: NACE classification code.
        unspsc_code: UNSPSC product code.
        vendor_name: Name of the capital goods vendor.
        vendor_country: Country of the vendor.
        eeio_database: EEIO database for factor lookup.
        margin_rate: Sector margin rate applied.
        fx_rate: Foreign exchange rate used.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier",
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the capital asset record",
    )
    amount: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="CapEx amount in original currency",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of the CapEx amount",
    )
    acquisition_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Year of acquisition for reporting",
    )
    spend_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CapEx converted to USD",
    )
    spend_producer_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CapEx adjusted to producer price after margin",
    )
    naics_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="NAICS 6-digit sector code",
    )
    nace_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="NACE classification code",
    )
    unspsc_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="UNSPSC product code",
    )
    vendor_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Name of the capital goods vendor",
    )
    vendor_country: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Country of the vendor",
    )
    eeio_database: EEIODatabase = Field(
        default=EEIODatabase.EPA_USEEIO,
        description="EEIO database for factor lookup",
    )
    margin_rate: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Sector margin rate applied",
    )
    fx_rate: Decimal = Field(
        default=Decimal("1.0"),
        gt=Decimal("0"),
        description="Foreign exchange rate used for conversion",
    )


# ---------------------------------------------------------------------------
# 3. PhysicalRecord
# ---------------------------------------------------------------------------


class PhysicalRecord(BaseModel):
    """Quantity-based input record for average-data calculation.

    A capital asset enriched with physical quantity data and
    material classification for emission factor lookup.

    Attributes:
        record_id: Unique record identifier.
        asset_id: Reference to the capital asset record.
        material_type: Material key for EF lookup.
        quantity: Physical quantity (e.g. mass, area, count).
        unit: Unit of the quantity (kg, m2, unit, kW, etc.).
        weight_kg: Total weight in kilograms.
        area_m2: Total area in square metres (buildings).
        asset_category: Asset category for EF selection.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier",
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the capital asset record",
    )
    material_type: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Material key for EF lookup",
    )
    quantity: Decimal = Field(
        default=Decimal("1"),
        gt=Decimal("0"),
        description="Physical quantity",
    )
    unit: str = Field(
        default="kg",
        max_length=50,
        description="Unit of the quantity (kg, m2, unit, kW)",
    )
    weight_kg: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Total weight in kilograms",
    )
    area_m2: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Total area in square metres (buildings)",
    )
    asset_category: Optional[AssetCategory] = Field(
        default=None,
        description="Asset category for EF selection",
    )


# ---------------------------------------------------------------------------
# 4. SupplierRecord
# ---------------------------------------------------------------------------


class SupplierRecord(BaseModel):
    """Supplier-specific input record for supplier-level calculation.

    Contains primary emission data from a specific capital goods
    supplier, including data source, allocation method, and
    verification status.

    Attributes:
        record_id: Unique record identifier.
        asset_id: Reference to the capital asset record.
        supplier_name: Name of the supplier.
        data_source: Source of the supplier emission data.
        ef_value: Emission factor value (kgCO2e per unit).
        ef_unit: Unit of the emission factor denominator.
        allocation_method: Allocation method for facility data.
        allocation_factor: Allocation factor (0-1).
        verification_status: Third-party verification status.
        epd_number: EPD registration number if applicable.
        reporting_year: Year the supplier data was reported.
        boundary: System boundary of supplier data.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier",
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the capital asset record",
    )
    supplier_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Name of the supplier",
    )
    data_source: SupplierDataSource = Field(
        ...,
        description="Source of the supplier emission data",
    )
    ef_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor value (kgCO2e per unit)",
    )
    ef_unit: str = Field(
        default="kgCO2e/unit",
        max_length=50,
        description="Unit of the emission factor denominator",
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.ECONOMIC,
        description="Allocation method for facility-level data",
    )
    allocation_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Allocation factor (0-1)",
    )
    verification_status: str = Field(
        default="unverified",
        max_length=50,
        description="Third-party verification status",
    )
    epd_number: Optional[str] = Field(
        default=None,
        max_length=100,
        description="EPD registration number if applicable",
    )
    reporting_year: Optional[int] = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Year the supplier data was reported",
    )
    boundary: str = Field(
        default="cradle_to_gate",
        max_length=50,
        description="System boundary (cradle_to_gate or cradle_to_grave)",
    )


# ---------------------------------------------------------------------------
# 5. SpendBasedResult
# ---------------------------------------------------------------------------


class SpendBasedResult(BaseModel):
    """Result of a spend-based emission calculation for one asset.

    Contains the calculated emissions using the EEIO method
    including intermediate values for currency conversion and
    margin removal steps.

    Attributes:
        record_id: Unique result identifier.
        asset_id: Reference to the source capital asset.
        spend_usd: CapEx spend converted to USD.
        eeio_factor: EEIO factor applied (kgCO2e per USD).
        emissions_kg_co2e: Total emissions in kgCO2e.
        co2: CO2 component in kgCO2e.
        ch4: CH4 component in kgCO2e.
        n2o: N2O component in kgCO2e.
        dqi_score: Composite data quality score.
        uncertainty_pct: Uncertainty percentage (+/-).
        method: Calculation method used.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier",
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source capital asset",
    )
    spend_usd: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CapEx spend converted to USD",
    )
    eeio_factor: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="EEIO factor applied (kgCO2e per USD)",
    )
    emissions_kg_co2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions in kgCO2e",
    )
    co2: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CO2 component in kgCO2e",
    )
    ch4: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 component in kgCO2e",
    )
    n2o: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="N2O component in kgCO2e",
    )
    dqi_score: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Composite data quality score (1-5)",
    )
    uncertainty_pct: Decimal = Field(
        default=Decimal("75.0"),
        ge=Decimal("0"),
        description="Uncertainty percentage (+/-)",
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.SPEND_BASED,
        description="Calculation method used",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash for audit trail",
    )


# ---------------------------------------------------------------------------
# 6. AverageDataResult
# ---------------------------------------------------------------------------


class AverageDataResult(BaseModel):
    """Result of an average-data emission calculation for one asset.

    Contains the calculated emissions using physical quantity
    multiplied by industry-average emission factors.

    Attributes:
        record_id: Unique result identifier.
        asset_id: Reference to the source capital asset.
        quantity: Physical quantity used in calculation.
        unit: Unit of the physical quantity.
        ef_value: Emission factor value applied.
        ef_source: Source of the emission factor.
        emissions_kg_co2e: Total emissions in kgCO2e.
        co2: CO2 component in kgCO2e.
        ch4: CH4 component in kgCO2e.
        n2o: N2O component in kgCO2e.
        transport_emissions: Additional transport emissions.
        dqi_score: Composite data quality score.
        uncertainty_pct: Uncertainty percentage (+/-).
        method: Calculation method used.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier",
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source capital asset",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Physical quantity used in calculation",
    )
    unit: str = Field(
        ...,
        max_length=50,
        description="Unit of the physical quantity",
    )
    ef_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor value applied",
    )
    ef_source: PhysicalEFSource = Field(
        ...,
        description="Source of the emission factor",
    )
    emissions_kg_co2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions in kgCO2e",
    )
    co2: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CO2 component in kgCO2e",
    )
    ch4: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 component in kgCO2e",
    )
    n2o: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="N2O component in kgCO2e",
    )
    transport_emissions: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Additional transport emissions in kgCO2e",
    )
    dqi_score: Decimal = Field(
        default=Decimal("3.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Composite data quality score (1-5)",
    )
    uncertainty_pct: Decimal = Field(
        default=Decimal("45.0"),
        ge=Decimal("0"),
        description="Uncertainty percentage (+/-)",
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.AVERAGE_DATA,
        description="Calculation method used",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash for audit trail",
    )


# ---------------------------------------------------------------------------
# 7. SupplierSpecificResult
# ---------------------------------------------------------------------------


class SupplierSpecificResult(BaseModel):
    """Result of a supplier-specific calculation for one asset.

    Contains the calculated emissions using primary supplier data,
    including allocation details and verification status.

    Attributes:
        record_id: Unique result identifier.
        asset_id: Reference to the source capital asset.
        supplier_name: Name of the supplier.
        data_source: Source of the supplier data.
        ef_value: Emission factor from supplier.
        allocation_method: Allocation method used.
        allocation_factor: Allocation factor applied (0-1).
        emissions_kg_co2e: Total emissions in kgCO2e.
        co2: CO2 component in kgCO2e.
        ch4: CH4 component in kgCO2e.
        n2o: N2O component in kgCO2e.
        verification_status: Verification status of supplier data.
        dqi_score: Composite data quality score.
        uncertainty_pct: Uncertainty percentage (+/-).
        method: Calculation method used.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier",
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source capital asset",
    )
    supplier_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Name of the supplier",
    )
    data_source: SupplierDataSource = Field(
        ...,
        description="Source of the supplier data",
    )
    ef_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor from supplier (kgCO2e/unit)",
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.ECONOMIC,
        description="Allocation method used",
    )
    allocation_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Allocation factor applied (0-1)",
    )
    emissions_kg_co2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions in kgCO2e",
    )
    co2: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CO2 component in kgCO2e",
    )
    ch4: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 component in kgCO2e",
    )
    n2o: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="N2O component in kgCO2e",
    )
    verification_status: str = Field(
        default="unverified",
        max_length=50,
        description="Verification status of supplier data",
    )
    dqi_score: Decimal = Field(
        default=Decimal("1.5"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Composite data quality score (1-5)",
    )
    uncertainty_pct: Decimal = Field(
        default=Decimal("20.0"),
        ge=Decimal("0"),
        description="Uncertainty percentage (+/-)",
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.SUPPLIER_SPECIFIC,
        description="Calculation method used",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash for audit trail",
    )


# ---------------------------------------------------------------------------
# 8. HybridResult
# ---------------------------------------------------------------------------


class HybridResult(BaseModel):
    """Aggregated result combining all calculation methods.

    The hybrid method combines supplier-specific, average-data,
    and spend-based results using the highest-quality data
    available for each capital asset.  Contains aggregated totals,
    method breakdown, and coverage analysis.

    CRITICAL: All emissions are reported in the year of
    acquisition.  No depreciation.

    Attributes:
        calculation_id: Unique identifier for this calculation.
        total_emissions_kg_co2e: Total Category 2 in kgCO2e.
        total_emissions_tco2e: Total Category 2 in tCO2e.
        spend_based_emissions_tco2e: From spend-based method.
        average_data_emissions_tco2e: From average-data method.
        supplier_specific_emissions_tco2e: From supplier method.
        spend_based_coverage_pct: CapEx % via spend-based.
        average_data_coverage_pct: CapEx % via average-data.
        supplier_specific_coverage_pct: CapEx % via supplier.
        total_coverage_pct: Total CapEx coverage.
        total_capex_usd: Total capital expenditure in USD.
        asset_count: Number of assets processed.
        spend_based_count: Assets using spend-based.
        average_data_count: Assets using average-data.
        supplier_specific_count: Assets using supplier data.
        excluded_count: Assets excluded (boundary checks).
        weighted_dqi: Emission-weighted DQI score.
        method_breakdown: Emissions by method Dict.
        provenance_hash: SHA-256 hash of aggregated result.
        timestamp: UTC timestamp of calculation.
        processing_time_ms: Processing duration in ms.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this calculation run",
    )
    total_emissions_kg_co2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total Category 2 emissions in kgCO2e",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total Category 2 emissions in tCO2e",
    )
    spend_based_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions from spend-based items in tCO2e",
    )
    average_data_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions from average-data items in tCO2e",
    )
    supplier_specific_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions from supplier items in tCO2e",
    )
    spend_based_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="CapEx percentage covered by spend-based",
    )
    average_data_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="CapEx percentage covered by average-data",
    )
    supplier_specific_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="CapEx percentage covered by supplier-specific",
    )
    total_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Total CapEx coverage percentage",
    )
    total_capex_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total capital expenditure in USD",
    )
    asset_count: int = Field(
        default=0,
        ge=0,
        description="Total number of assets processed",
    )
    spend_based_count: int = Field(
        default=0,
        ge=0,
        description="Assets using spend-based method",
    )
    average_data_count: int = Field(
        default=0,
        ge=0,
        description="Assets using average-data method",
    )
    supplier_specific_count: int = Field(
        default=0,
        ge=0,
        description="Assets using supplier-specific method",
    )
    excluded_count: int = Field(
        default=0,
        ge=0,
        description="Assets excluded by boundary checks",
    )
    weighted_dqi: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Emission-weighted composite DQI score",
    )
    method_breakdown: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by method (tCO2e)",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the aggregated result",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of calculation",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )


# ---------------------------------------------------------------------------
# 9. EEIOFactor
# ---------------------------------------------------------------------------


class EEIOFactor(BaseModel):
    """An EEIO emission factor entry for capital goods sectors.

    Maps a NAICS sector code to an emission factor expressed in
    kgCO2e per unit of economic output (USD).

    Attributes:
        naics_code: NAICS 6-digit sector code.
        description: Human-readable sector name.
        factor_kg_co2e_per_usd: Factor in kgCO2e per USD.
        source: EEIO database source.
        year: Base year for the economic data.
        region: Geographic region.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    naics_code: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="NAICS 6-digit sector code",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable sector name",
    )
    factor_kg_co2e_per_usd: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor in kgCO2e per USD",
    )
    source: EEIODatabase = Field(
        default=EEIODatabase.EPA_USEEIO,
        description="EEIO database source",
    )
    year: int = Field(
        default=2021,
        ge=2000,
        le=2100,
        description="Base year for the economic data",
    )
    region: str = Field(
        default="US",
        max_length=20,
        description="Geographic region",
    )


# ---------------------------------------------------------------------------
# 10. PhysicalEF
# ---------------------------------------------------------------------------


class PhysicalEF(BaseModel):
    """A physical emission factor for capital goods materials.

    Maps a material or asset type to a cradle-to-gate emission
    factor in kgCO2e per kg (or kgCO2e per unit).

    Attributes:
        material_type: Unique key for the material or asset type.
        factor_kg_co2e_per_unit: EF in kgCO2e per unit.
        unit: Denominator unit (kg, unit, kW, m2, etc.).
        source: Source database for the factor.
        region: Geographic region the factor applies to.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    material_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique key for the material or asset type",
    )
    factor_kg_co2e_per_unit: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor in kgCO2e per unit",
    )
    unit: str = Field(
        default="kg",
        max_length=50,
        description="Denominator unit (kg, unit, kW, m2)",
    )
    source: PhysicalEFSource = Field(
        default=PhysicalEFSource.ICE_DATABASE,
        description="Source database for the factor",
    )
    region: str = Field(
        default="GLOBAL",
        max_length=20,
        description="Geographic region the factor applies to",
    )


# ---------------------------------------------------------------------------
# 11. SupplierEF
# ---------------------------------------------------------------------------


class SupplierEF(BaseModel):
    """A supplier-specific emission factor for capital goods.

    Represents emission data provided by a specific capital goods
    supplier for their products or facilities.

    Attributes:
        supplier_name: Name of the supplier.
        product_type: Product or asset type name.
        ef_value: Emission factor value (kgCO2e per unit).
        ef_unit: Denominator unit.
        source: Source of the supplier data.
        verification: Verification status.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    supplier_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Name of the supplier",
    )
    product_type: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product or asset type name",
    )
    ef_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor value (kgCO2e per unit)",
    )
    ef_unit: str = Field(
        default="kgCO2e/unit",
        max_length=50,
        description="Denominator unit",
    )
    source: SupplierDataSource = Field(
        ...,
        description="Source of the supplier data",
    )
    verification: str = Field(
        default="unverified",
        max_length=50,
        description="Verification status",
    )


# ---------------------------------------------------------------------------
# 12. DQIAssessment
# ---------------------------------------------------------------------------


class DQIAssessment(BaseModel):
    """Data quality indicator assessment for a calculation result.

    Scores data quality across the five GHG Protocol dimensions
    and computes a composite score (arithmetic mean).  Lower
    scores indicate higher quality.

    Attributes:
        asset_id: Reference to the capital asset.
        calculation_method: Calculation method used.
        temporal_score: Temporal representativeness (1-5).
        geographical_score: Geographical representativeness (1-5).
        technological_score: Technological representativeness (1-5).
        completeness_score: Data completeness (1-5).
        reliability_score: Data reliability (1-5).
        composite_score: Arithmetic mean of all five scores.
        quality_tier: Qualitative quality tier label.
        uncertainty_factor: Pedigree uncertainty factor.
        findings: List of findings and recommendations.
        ef_hierarchy_level: EF hierarchy level used (1-8).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    asset_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the capital asset",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Calculation method used",
    )
    temporal_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Temporal representativeness score (1-5)",
    )
    geographical_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Geographical representativeness score (1-5)",
    )
    technological_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Technological representativeness score (1-5)",
    )
    completeness_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Data completeness score (1-5)",
    )
    reliability_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Data reliability score (1-5)",
    )
    composite_score: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Arithmetic mean of all five scores",
    )
    quality_tier: str = Field(
        default="",
        max_length=50,
        description="Qualitative quality tier label",
    )
    uncertainty_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("1.0"),
        description="Combined pedigree uncertainty factor",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="List of findings and recommendations",
    )
    ef_hierarchy_level: int = Field(
        default=8,
        ge=1,
        le=8,
        description="EF hierarchy level used (1=best, 8=worst)",
    )


# ---------------------------------------------------------------------------
# 13. AssetClassification
# ---------------------------------------------------------------------------


class AssetClassification(BaseModel):
    """Classification result for a capital asset.

    Determines whether an expenditure qualifies as a capital good
    (Category 2) or should be treated as an operating expense
    (Category 1), and assigns asset category and NAICS codes.

    Attributes:
        asset_id: Reference to the asset record.
        category: Assigned top-level asset category.
        subcategory: Assigned sub-category.
        naics_code: Resolved NAICS 6-digit code.
        nace_code: Resolved NACE code.
        is_capital: Whether classified as capital good.
        capitalization_met: Whether capitalization threshold met.
        classification_confidence: Confidence percentage.
        classification_reason: Explanation of classification.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    asset_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the asset record",
    )
    category: AssetCategory = Field(
        ...,
        description="Assigned top-level asset category",
    )
    subcategory: Optional[AssetSubCategory] = Field(
        default=None,
        description="Assigned sub-category",
    )
    naics_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="Resolved NAICS 6-digit code",
    )
    nace_code: Optional[str] = Field(
        default=None,
        max_length=10,
        description="Resolved NACE code",
    )
    is_capital: bool = Field(
        ...,
        description="Whether classified as capital good (Cat 2)",
    )
    capitalization_met: bool = Field(
        ...,
        description="Whether capitalization threshold is met",
    )
    classification_confidence: Decimal = Field(
        default=Decimal("100.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Confidence percentage of classification",
    )
    classification_reason: str = Field(
        default="",
        max_length=2000,
        description="Explanation of classification decision",
    )


# ---------------------------------------------------------------------------
# 14. CapitalizationThreshold
# ---------------------------------------------------------------------------


class CapitalizationThreshold(BaseModel):
    """Capitalization threshold configuration.

    Defines the monetary and useful-life thresholds for
    classifying expenditures as capital goods (PP&E) vs.
    operating expenses.

    Attributes:
        policy: Accounting capitalization policy.
        threshold_amount: Minimum CapEx amount for capitalization.
        currency: Currency of the threshold amount.
        useful_life_min_years: Minimum useful life for capitalization.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy: CapitalizationPolicy = Field(
        ...,
        description="Accounting capitalization policy",
    )
    threshold_amount: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Minimum CapEx amount for capitalization",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of the threshold amount",
    )
    useful_life_min_years: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Minimum useful life in years for capitalization",
    )


# ---------------------------------------------------------------------------
# 15. UsefulLifeRange
# ---------------------------------------------------------------------------


class UsefulLifeRange(BaseModel):
    """Useful life range for an asset category or sub-category.

    Used for validation and default assignment when useful life
    is not specified on the asset record.

    Attributes:
        asset_category: Asset category or sub-category key.
        min_years: Minimum useful life in years.
        max_years: Maximum useful life in years.
        default_years: Default useful life in years.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    asset_category: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Asset category or sub-category key",
    )
    min_years: int = Field(
        ...,
        ge=1,
        le=100,
        description="Minimum useful life in years",
    )
    max_years: int = Field(
        ...,
        ge=1,
        le=100,
        description="Maximum useful life in years",
    )
    default_years: int = Field(
        ...,
        ge=1,
        le=100,
        description="Default useful life in years",
    )

    @field_validator("max_years")
    @classmethod
    def _max_gte_min(cls, v: int, info: Any) -> int:
        """Validate max_years >= min_years."""
        min_y = info.data.get("min_years")
        if min_y is not None and v < min_y:
            raise ValueError(
                f"max_years ({v}) must be >= min_years ({min_y})"
            )
        return v

    @field_validator("default_years")
    @classmethod
    def _default_in_range(cls, v: int, info: Any) -> int:
        """Validate default_years is within [min, max] range."""
        min_y = info.data.get("min_years")
        max_y = info.data.get("max_years")
        if min_y is not None and v < min_y:
            raise ValueError(
                f"default_years ({v}) must be >= min_years ({min_y})"
            )
        if max_y is not None and v > max_y:
            raise ValueError(
                f"default_years ({v}) must be <= max_years ({max_y})"
            )
        return v


# ---------------------------------------------------------------------------
# 16. DepreciationContext
# ---------------------------------------------------------------------------


class DepreciationContext(BaseModel):
    """Year-over-year CapEx variance context.

    NOTE: This is NOT for depreciation of emissions.  GHG Protocol
    requires 100 % of cradle-to-gate emissions to be reported in
    the year of acquisition.  This model provides YoY variance
    context so users understand why Category 2 emissions may spike
    in major CapEx years.

    Attributes:
        acquisition_year: The reporting year.
        total_capex: Total CapEx for the year in USD.
        rolling_avg_capex: 3-year rolling average CapEx in USD.
        volatility_ratio: Ratio of current year to rolling avg.
        is_major_capex_year: Whether CapEx exceeds 1.5x avg.
        context_note: Explanatory note for reporting narrative.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    acquisition_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="The reporting year",
    )
    total_capex: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CapEx for the year in USD",
    )
    rolling_avg_capex: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="3-year rolling average CapEx in USD",
    )
    volatility_ratio: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        description="Ratio of current year CapEx to rolling avg",
    )
    is_major_capex_year: bool = Field(
        default=False,
        description="Whether CapEx exceeds 1.5x rolling average",
    )
    context_note: str = Field(
        default="",
        max_length=2000,
        description="Explanatory note for reporting narrative",
    )


# ---------------------------------------------------------------------------
# 17. MaterialityItem
# ---------------------------------------------------------------------------


class MaterialityItem(BaseModel):
    """A single item in the hot-spot materiality analysis.

    Represents one asset category or supplier ranked by emission
    contribution for Pareto analysis and materiality quadrant
    classification.

    Attributes:
        asset_category: Category or supplier identifier.
        emissions_kg_co2e: Emissions in kgCO2e.
        pct_of_total: Percentage of total Category 2 emissions.
        cumulative_pct: Cumulative % for Pareto ranking.
        is_material: Whether above materiality threshold.
        quadrant: Materiality quadrant classification.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    asset_category: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Category or supplier identifier",
    )
    emissions_kg_co2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emissions in kgCO2e",
    )
    pct_of_total: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of total Category 2 emissions",
    )
    cumulative_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Cumulative percentage for Pareto ranking",
    )
    is_material: bool = Field(
        default=False,
        description="Whether above materiality threshold",
    )
    quadrant: str = Field(
        default="low_priority",
        max_length=50,
        description=(
            "Materiality quadrant: prioritize, monitor, "
            "improve_data, or low_priority"
        ),
    )


# ---------------------------------------------------------------------------
# 18. CoverageReport
# ---------------------------------------------------------------------------


class CoverageReport(BaseModel):
    """Method coverage analysis for the Category 2 inventory.

    Summarizes the breakdown of CapEx and assets by calculation
    method, including coverage percentages and gap identification.

    Attributes:
        total_assets: Total number of capital assets.
        covered_assets: Number of assets with calculations.
        coverage_pct: Percentage of assets covered.
        by_method: Asset count and CapEx by method.
        uncovered_capex_usd: CapEx not covered by any method.
        gap_categories: Asset categories with no coverage.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    total_assets: int = Field(
        ...,
        ge=0,
        description="Total number of capital assets",
    )
    covered_assets: int = Field(
        default=0,
        ge=0,
        description="Number of assets with calculations",
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of assets covered",
    )
    by_method: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description=(
            "Asset count and CapEx by method, e.g. "
            "{'spend_based': {'count': 10, 'capex_usd': 1000}}"
        ),
    )
    uncovered_capex_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CapEx not covered by any method",
    )
    gap_categories: List[str] = Field(
        default_factory=list,
        description="Asset categories with no coverage",
    )


# ---------------------------------------------------------------------------
# 19. ComplianceRequirement
# ---------------------------------------------------------------------------


class ComplianceRequirement(BaseModel):
    """A single compliance requirement for a regulatory framework.

    Represents one disclosure or data requirement that must be
    satisfied for compliance with a specific framework.

    Attributes:
        framework: The regulatory framework.
        requirement_id: Machine-readable requirement identifier.
        description: Human-readable description.
        mandatory: Whether this requirement is mandatory.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    framework: ComplianceFramework = Field(
        ...,
        description="The regulatory framework",
    )
    requirement_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Machine-readable requirement identifier",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Human-readable description",
    )
    mandatory: bool = Field(
        default=True,
        description="Whether this requirement is mandatory",
    )


# ---------------------------------------------------------------------------
# 20. ComplianceCheckResult
# ---------------------------------------------------------------------------


class ComplianceCheckResult(BaseModel):
    """Result of a compliance check against one framework.

    Aggregates individual requirement results into an overall
    compliance status for a specific framework.

    Attributes:
        framework: The regulatory framework checked.
        status: Overall compliance status.
        requirements_met: Number of requirements met.
        requirements_total: Total number of requirements.
        gaps: List of unmet requirement descriptions.
        recommendations: Improvement recommendations.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    framework: ComplianceFramework = Field(
        ...,
        description="The regulatory framework checked",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status",
    )
    requirements_met: int = Field(
        default=0,
        ge=0,
        description="Number of requirements met",
    )
    requirements_total: int = Field(
        default=0,
        ge=0,
        description="Total number of requirements checked",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="List of unmet requirement descriptions",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations",
    )


# ---------------------------------------------------------------------------
# 21. CalculationRequest
# ---------------------------------------------------------------------------


class CalculationRequest(BaseModel):
    """Main calculation request for Category 2 capital goods emissions.

    Primary input to the Capital Goods calculation pipeline.
    Contains asset records, configuration for calculation methods,
    and options for compliance checking and export.

    Attributes:
        request_id: Unique request identifier.
        tenant_id: Tenant identifier for multi-tenancy.
        asset_records: List of capital asset records.
        calculation_method: Preferred calculation method.
        gwp_source: IPCC AR version for GWP values.
        base_currency: Base currency for CapEx normalization.
        reporting_year: Reporting year for emissions allocation.
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        facility_id: Optional facility filter.
        compliance_frameworks: Frameworks to check against.
        include_uncertainty: Whether to quantify uncertainty.
        include_dqi: Whether to score data quality.
        include_hotspot: Whether to run hot-spot analysis.
        capitalization_threshold: Capitalization threshold config.
        export_format: Requested export format.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy",
    )
    asset_records: List[CapitalAssetRecord] = Field(
        ...,
        min_length=1,
        description="List of capital asset records to calculate",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.HYBRID,
        description="Preferred calculation method",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR5,
        description="IPCC AR version for GWP values",
    )
    base_currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Base currency for CapEx normalization",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year for emissions allocation",
    )
    period_start: date = Field(
        ...,
        description="Start of the reporting period",
    )
    period_end: date = Field(
        ...,
        description="End of the reporting period",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional facility filter",
    )
    compliance_frameworks: Optional[List[ComplianceFramework]] = Field(
        default=None,
        description="Frameworks to check compliance against",
    )
    include_uncertainty: bool = Field(
        default=True,
        description="Whether to quantify uncertainty",
    )
    include_dqi: bool = Field(
        default=True,
        description="Whether to score data quality",
    )
    include_hotspot: bool = Field(
        default=True,
        description="Whether to run hot-spot analysis",
    )
    capitalization_threshold: Optional[CapitalizationThreshold] = Field(
        default=None,
        description="Capitalization threshold configuration",
    )
    export_format: Optional[ExportFormat] = Field(
        default=None,
        description="Requested export format",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

    @field_validator("asset_records")
    @classmethod
    def _validate_asset_count(
        cls, v: List[CapitalAssetRecord]
    ) -> List[CapitalAssetRecord]:
        """Validate that assets do not exceed maximum."""
        if len(v) > MAX_ASSET_RECORDS:
            raise ValueError(
                f"Maximum {MAX_ASSET_RECORDS} assets per "
                f"request, got {len(v)}"
            )
        return v

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: date, info: Any) -> date:
        """Validate that period_end is on or after period_start."""
        start = info.data.get("period_start")
        if start is not None and v < start:
            raise ValueError(
                f"period_end ({v}) must be on or after "
                f"period_start ({start})"
            )
        return v

    @field_validator("compliance_frameworks")
    @classmethod
    def _validate_frameworks_count(
        cls, v: Optional[List[ComplianceFramework]]
    ) -> Optional[List[ComplianceFramework]]:
        """Validate that frameworks do not exceed maximum."""
        if v is not None and len(v) > MAX_FRAMEWORKS:
            raise ValueError(
                f"Maximum {MAX_FRAMEWORKS} frameworks per "
                f"request, got {len(v)}"
            )
        return v


# ---------------------------------------------------------------------------
# 22. BatchRequest
# ---------------------------------------------------------------------------


class BatchRequest(BaseModel):
    """Request to perform calculations across multiple periods.

    Enables batch processing of Category 2 calculations for
    multiple reporting periods in a single request.

    Attributes:
        batch_id: Unique batch job identifier.
        tenant_id: Tenant identifier for multi-tenancy.
        requests: List of individual calculation requests.
        parallel: Whether to process periods in parallel.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch job identifier",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy",
    )
    requests: List[CalculationRequest] = Field(
        ...,
        min_length=1,
        description="List of individual calculation requests",
    )
    parallel: bool = Field(
        default=False,
        description="Whether to process periods in parallel",
    )

    @field_validator("requests")
    @classmethod
    def _validate_requests_count(
        cls, v: List[CalculationRequest]
    ) -> List[CalculationRequest]:
        """Validate that requests do not exceed maximum."""
        if len(v) > MAX_BATCH_PERIODS:
            raise ValueError(
                f"Maximum {MAX_BATCH_PERIODS} requests per "
                f"batch, got {len(v)}"
            )
        return v


# ---------------------------------------------------------------------------
# 23. CalculationResult
# ---------------------------------------------------------------------------


class CalculationResult(BaseModel):
    """Complete output of a Category 2 emission calculation.

    The primary output of the calculation pipeline, containing
    all method results, hybrid aggregation, compliance results,
    aggregation summaries, hot-spot analysis, and provenance.

    CRITICAL: All emissions are reported in the year of
    acquisition.  No depreciation applied.

    Attributes:
        calculation_id: Unique calculation identifier.
        request_id: Reference to the originating request.
        tenant_id: Tenant identifier.
        status: Calculation status.
        spend_based_results: Spend-based line-item results.
        average_data_results: Average-data line-item results.
        supplier_specific_results: Supplier line-item results.
        hybrid_result: Aggregated hybrid result.
        compliance_results: Compliance check per framework.
        aggregation: Multi-dimension aggregation summary.
        hot_spots: Hot-spot / Pareto analysis.
        depreciation_context: YoY CapEx variance context.
        dqi_assessments: DQI assessments per asset.
        coverage_report: Method coverage analysis.
        asset_classifications: Asset classification results.
        provenance_hash: SHA-256 hash over entire result.
        timestamp: UTC timestamp of calculation.
        processing_time_ms: Processing duration in ms.
        pipeline_stages_completed: Completed pipeline stages.
        warnings: Warning messages.
        errors: Error messages.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation identifier",
    )
    request_id: str = Field(
        default="",
        max_length=200,
        description="Reference to the originating request",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier",
    )
    status: BatchStatus = Field(
        default=BatchStatus.PENDING,
        description="Calculation status",
    )
    spend_based_results: List[SpendBasedResult] = Field(
        default_factory=list,
        description="Spend-based line-item results",
    )
    average_data_results: List[AverageDataResult] = Field(
        default_factory=list,
        description="Average-data line-item results",
    )
    supplier_specific_results: List[SupplierSpecificResult] = Field(
        default_factory=list,
        description="Supplier-specific line-item results",
    )
    hybrid_result: Optional[HybridResult] = Field(
        default=None,
        description="Aggregated hybrid result with totals",
    )
    compliance_results: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="Compliance check results per framework",
    )
    aggregation: Optional[AggregationResult] = Field(
        default=None,
        description="Multi-dimension aggregation summary",
    )
    hot_spots: Optional[HotSpotAnalysis] = Field(
        default=None,
        description="Hot-spot / Pareto analysis",
    )
    depreciation_context: Optional[DepreciationContext] = Field(
        default=None,
        description="YoY CapEx variance context (NOT emission depreciation)",
    )
    dqi_assessments: List[DQIAssessment] = Field(
        default_factory=list,
        description="DQI assessments per asset",
    )
    coverage_report: Optional[CoverageReport] = Field(
        default=None,
        description="Method coverage analysis",
    )
    asset_classifications: List[AssetClassification] = Field(
        default_factory=list,
        description="Asset classification results",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash over the entire result",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of calculation completion",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )
    pipeline_stages_completed: List[str] = Field(
        default_factory=list,
        description="List of completed pipeline stage names",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )


# ---------------------------------------------------------------------------
# 24. AggregationResult
# ---------------------------------------------------------------------------


class AggregationResult(BaseModel):
    """Multi-dimension aggregation of Category 2 results.

    Provides breakdowns by asset category, by calculation method,
    by supplier, and by reporting period.

    Attributes:
        aggregation_id: Unique aggregation identifier.
        total_emissions_tco2e: Total aggregated emissions.
        total_capex_usd: Total CapEx in USD.
        by_category: Emissions by asset category (tCO2e).
        by_method: Emissions by calculation method (tCO2e).
        by_supplier: Emissions by supplier (tCO2e).
        by_period: Emissions by reporting period (tCO2e).
        by_facility: Emissions by facility (tCO2e).
        intensity_per_capex: tCO2e per $M CapEx.
        intensity_per_revenue: tCO2e per $M revenue.
        provenance_hash: SHA-256 hash of the aggregation.
        timestamp: UTC timestamp of aggregation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    aggregation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique aggregation identifier",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total aggregated emissions in tCO2e",
    )
    total_capex_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total CapEx in USD",
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by asset category (tCO2e)",
    )
    by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by calculation method (tCO2e)",
    )
    by_supplier: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by supplier (tCO2e)",
    )
    by_period: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by reporting period (tCO2e)",
    )
    by_facility: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by facility (tCO2e)",
    )
    intensity_per_capex: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Emission intensity (tCO2e per $M CapEx)",
    )
    intensity_per_revenue: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Emission intensity (tCO2e per $M revenue)",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the aggregation",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of aggregation",
    )


# ---------------------------------------------------------------------------
# 25. HotSpotAnalysis
# ---------------------------------------------------------------------------


class HotSpotAnalysis(BaseModel):
    """Pareto hot-spot analysis of Category 2 emission contributors.

    Identifies top asset categories and suppliers by emission
    contribution using 80/20 Pareto analysis and assigns
    materiality quadrant classifications.

    Attributes:
        calculation_id: Reference to the calculation.
        total_emissions_tco2e: Total emissions analysed.
        top_assets: Top N assets by emission contribution.
        pareto_items: Ranked materiality items.
        materiality_quadrants: Items by quadrant.
        recommendations: Prioritization recommendations.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the calculation",
    )
    total_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions analysed in tCO2e",
    )
    top_assets: List[MaterialityItem] = Field(
        default_factory=list,
        description="Top N assets by emission contribution",
    )
    pareto_items: List[MaterialityItem] = Field(
        default_factory=list,
        description="All items ranked by Pareto contribution",
    )
    materiality_quadrants: Dict[str, List[MaterialityItem]] = Field(
        default_factory=dict,
        description="Items grouped by materiality quadrant",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritization recommendations",
    )

    @field_validator("top_assets")
    @classmethod
    def _validate_top_assets_count(
        cls, v: List[MaterialityItem]
    ) -> List[MaterialityItem]:
        """Validate that top assets do not exceed maximum."""
        if len(v) > MAX_HOTSPOT_ITEMS:
            raise ValueError(
                f"Maximum {MAX_HOTSPOT_ITEMS} hot-spot items, "
                f"got {len(v)}"
            )
        return v

    @field_validator("pareto_items")
    @classmethod
    def _validate_pareto_count(
        cls, v: List[MaterialityItem]
    ) -> List[MaterialityItem]:
        """Validate that pareto items do not exceed maximum."""
        if len(v) > MAX_HOTSPOT_ITEMS:
            raise ValueError(
                f"Maximum {MAX_HOTSPOT_ITEMS} pareto items, "
                f"got {len(v)}"
            )
        return v


# =============================================================================
# Type Aliases (backward-compatible names)
# =============================================================================

#: Alias for MaterialityItem (backward compatibility).
HotSpotItem = MaterialityItem

#: Alias for ComplianceRequirement (backward compatibility).
ComplianceRule = ComplianceRequirement


# =============================================================================
# __all__ -- Public API
# =============================================================================

__all__ = [
    # Module-level constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "MAX_ASSET_RECORDS",
    "MAX_BATCH_PERIODS",
    "MAX_FACILITIES",
    "MAX_SUPPLIERS",
    "MAX_FRAMEWORKS",
    "MAX_REQUIREMENTS_PER_FRAMEWORK",
    "MAX_HOTSPOT_ITEMS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DECIMAL_INF",
    "DECIMAL_PLACES",
    "ZERO",
    "ONE",
    "ONE_HUNDRED",
    "ONE_THOUSAND",
    # Enumerations (20)
    "CalculationMethod",
    "AssetCategory",
    "AssetSubCategory",
    "SpendClassificationSystem",
    "EEIODatabase",
    "PhysicalEFSource",
    "SupplierDataSource",
    "AllocationMethod",
    "CurrencyCode",
    "DQIDimension",
    "DQIScore",
    "UncertaintyMethod",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "ExportFormat",
    "BatchStatus",
    "GWPSource",
    "EmissionGas",
    "CapitalizationPolicy",
    # Constant tables (13)
    "GWP_VALUES",
    "DQI_SCORE_VALUES",
    "DQI_QUALITY_TIERS",
    "UNCERTAINTY_RANGES",
    "COVERAGE_THRESHOLDS",
    "EF_HIERARCHY_PRIORITY",
    "PEDIGREE_UNCERTAINTY_FACTORS",
    "CURRENCY_EXCHANGE_RATES",
    "CAPITAL_SECTOR_MARGIN_PERCENTAGES",
    "CAPITAL_EEIO_EMISSION_FACTORS",
    "CAPITAL_PHYSICAL_EMISSION_FACTORS",
    "ASSET_USEFUL_LIFE_RANGES",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    # Data models (25)
    "CapitalAssetRecord",
    "CapExSpendRecord",
    "PhysicalRecord",
    "SupplierRecord",
    "SpendBasedResult",
    "AverageDataResult",
    "SupplierSpecificResult",
    "HybridResult",
    "EEIOFactor",
    "PhysicalEF",
    "SupplierEF",
    "DQIAssessment",
    "AssetClassification",
    "CapitalizationThreshold",
    "UsefulLifeRange",
    "DepreciationContext",
    "MaterialityItem",
    "CoverageReport",
    "ComplianceRequirement",
    "ComplianceCheckResult",
    "CalculationRequest",
    "BatchRequest",
    "CalculationResult",
    "AggregationResult",
    "HotSpotAnalysis",
    # Type aliases
    "HotSpotItem",
    "ComplianceRule",
    # Helper
    "_utcnow",
]
