# -*- coding: utf-8 -*-
"""
Fuel & Energy Activities Agent Data Models - AGENT-MRV-016

Pydantic v2 data models for the Fuel & Energy Activities Agent SDK
covering GHG Protocol Scope 3 Category 3 emissions from fuel- and
energy-related activities not included in Scope 1 or Scope 2.

Category 3 includes four distinct sub-activities:
  3a - Upstream emissions of purchased fuels (well-to-tank / WTT)
  3b - Upstream emissions of purchased electricity (generation lifecycle)
  3c - Transmission & distribution losses (T&D line losses)
  3d - Generation of purchased electricity sold to end users (utilities)

Key design rules:
  - WTT emission factors are deterministic lookups (zero hallucination).
  - All arithmetic uses Python Decimal for reproducibility.
  - Provenance tracking via SHA-256 hashing at every pipeline stage.
  - Dual accounting support: location-based AND market-based for 3b/3c.
  - T&D loss factors from IEA, World Bank, EPA eGRID, or custom.
  - Supplier-specific data from EPDs, PCFs, MiQ certificates, PPAs.
  - Seven compliance frameworks: GHG Protocol, CSRD, CDP, SBTi,
    SB 253, GRI 305, ISO 14064.

Enumerations (22):
    CalculationMethod, FuelType, FuelCategory, EnergyType,
    ActivityType, WTTFactorSource, GridRegionType, TDLossSource,
    SupplierDataSource, AllocationMethod, CurrencyCode,
    DQIDimension, DQIScore, UncertaintyMethod, ComplianceFramework,
    ComplianceStatus, PipelineStage, ReportFormat, BatchStatus,
    GWPSource, EmissionGas, AccountingMethod

Constants (13):
    GWP_VALUES, WTT_FUEL_EMISSION_FACTORS, UPSTREAM_ELECTRICITY_FACTORS,
    TD_LOSS_FACTORS, EGRID_TD_LOSS_FACTORS, FUEL_HEATING_VALUES,
    FUEL_DENSITY_FACTORS, DQI_SCORE_VALUES, DQI_QUALITY_TIERS,
    UNCERTAINTY_RANGES, COVERAGE_THRESHOLDS, EF_HIERARCHY_PRIORITY,
    FRAMEWORK_REQUIRED_DISCLOSURES

Data Models (25):
    FuelConsumptionRecord, ElectricityConsumptionRecord,
    WTTEmissionFactor, UpstreamElectricityFactor, TDLossFactor,
    SupplierFuelData, Activity3aResult, Activity3bResult,
    Activity3cResult, Activity3dResult, CalculationResult,
    GasBreakdown, DQIAssessment, UncertaintyResult,
    ComplianceCheckResult, ComplianceFinding, PipelineResult,
    BatchRequest, BatchResult, AggregationResult, ExportRequest,
    MaterialityResult, HotSpotResult, YoYDecomposition,
    ProvenanceRecord

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-S3-003)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, field_validator

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module-level Constants
# ---------------------------------------------------------------------------

#: Agent identifier for registry integration.
AGENT_ID: str = "GL-MRV-S3-003"

#: Agent component identifier.
AGENT_COMPONENT: str = "AGENT-MRV-016"

#: Service version string.
VERSION: str = "1.0.0"

#: Database table prefix for all Fuel & Energy Activities tables.
TABLE_PREFIX: str = "gl_fea_"

#: Maximum number of fuel consumption records per calculation request.
MAX_FUEL_RECORDS: int = 100_000

#: Maximum number of electricity consumption records per request.
MAX_ELECTRICITY_RECORDS: int = 100_000

#: Maximum number of periods in a batch request.
MAX_BATCH_PERIODS: int = 120

#: Maximum number of facilities per aggregation.
MAX_FACILITIES: int = 50_000

#: Maximum number of suppliers per request.
MAX_SUPPLIERS: int = 10_000

#: Maximum number of frameworks per compliance check.
MAX_FRAMEWORKS: int = 20

#: Maximum number of compliance findings per framework.
MAX_FINDINGS_PER_FRAMEWORK: int = 200

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
# Enumerations (22)
# =============================================================================

class CalculationMethod(str, Enum):
    """GHG Protocol Scope 3 Category 3 calculation methods.

    The GHG Protocol Technical Guidance defines four methods for
    Category 3 fuel- and energy-related activities, listed from
    most to least accurate.

    SUPPLIER_SPECIFIC: Uses primary WTT / upstream data from fuel
        or electricity suppliers.  Highest accuracy (+/- 5-15 %).
    AVERAGE_DATA: Uses published average WTT emission factors
        from databases (DEFRA, EPA, IEA, ecoinvent).  Medium
        accuracy (+/- 15-30 %).
    SPEND_BASED: Estimates upstream emissions by multiplying fuel
        or electricity spend by EEIO factors.  Lowest accuracy
        (+/- 30-60 %).
    HYBRID: Combines supplier-specific and average-data methods
        using the best available data for each fuel / grid region.
    """

    SUPPLIER_SPECIFIC = "supplier_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"

class FuelType(str, Enum):
    """Fuel types for Activity 3a upstream (WTT) calculations.

    Twenty-five fuel types covering fossil fuels, biofuels, and
    waste-derived fuels.  Each maps to a set of well-to-tank
    emission factors (CO2, CH4, N2O) expressed in kgCO2e per kWh
    of fuel energy content.
    """

    # Fossil -- gaseous
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    PROPANE = "propane"

    # Fossil -- liquid
    DIESEL = "diesel"
    PETROL_GASOLINE = "petrol_gasoline"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    KEROSENE = "kerosene"
    JET_FUEL = "jet_fuel"
    PETROLEUM_COKE = "petroleum_coke"
    WASTE_OIL = "waste_oil"

    # Fossil -- solid
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    COAL_LIGNITE = "coal_lignite"
    COAL_ANTHRACITE = "coal_anthracite"
    PEAT = "peat"
    MSW = "msw"

    # Biofuels
    ETHANOL = "ethanol"
    BIODIESEL = "biodiesel"
    BIOGAS = "biogas"
    HVO = "hvo"

    # Biomass
    WOOD_PELLETS = "wood_pellets"
    BIOMASS_SOLID = "biomass_solid"
    BIOMASS_LIQUID = "biomass_liquid"
    LANDFILL_GAS = "landfill_gas"

class FuelCategory(str, Enum):
    """Broad fuel classification for reporting roll-ups.

    FOSSIL: Conventional fossil fuels (coal, oil, gas).
    BIOFUEL: First- and second-generation biofuels.
    WASTE_DERIVED: Municipal solid waste and waste oils.
    """

    FOSSIL = "fossil"
    BIOFUEL = "biofuel"
    WASTE_DERIVED = "waste_derived"

class EnergyType(str, Enum):
    """Types of purchased energy for Activity 3b and 3c.

    ELECTRICITY: Purchased grid or off-grid electricity.
    STEAM: Purchased steam from district or industrial sources.
    HEATING: Purchased district heating.
    COOLING: Purchased district cooling.
    """

    ELECTRICITY = "electricity"
    STEAM = "steam"
    HEATING = "heating"
    COOLING = "cooling"

class ActivityType(str, Enum):
    """Sub-activities within Scope 3 Category 3.

    ACTIVITY_3A: Upstream emissions of purchased fuels (WTT).
    ACTIVITY_3B: Upstream emissions of purchased electricity.
    ACTIVITY_3C: Transmission and distribution losses.
    ACTIVITY_3D: Generation of electricity sold to end users
        (applicable to utilities / energy resellers only).
    """

    ACTIVITY_3A = "activity_3a"
    ACTIVITY_3B = "activity_3b"
    ACTIVITY_3C = "activity_3c"
    ACTIVITY_3D = "activity_3d"

class WTTFactorSource(str, Enum):
    """Sources of well-to-tank emission factors for fuels.

    DEFRA: UK DEFRA/DESNZ conversion factors (annual).
    EPA: US EPA emission factor hub.
    IEA: International Energy Agency lifecycle data.
    ECOINVENT: ecoinvent v3.11 LCA database.
    GREET: Argonne National Laboratory GREET model.
    JEC: JEC Well-to-Wheels (EU Joint Research Centre).
    CUSTOM: User-defined or organisation-specific factors.
    """

    DEFRA = "defra"
    EPA = "epa"
    IEA = "iea"
    ECOINVENT = "ecoinvent"
    GREET = "greet"
    JEC = "jec"
    CUSTOM = "custom"

class GridRegionType(str, Enum):
    """Types of electricity grid regions for factor resolution.

    COUNTRY: National-level grid average.
    EGRID_SUBREGION: US EPA eGRID subregion.
    EU_MEMBER_STATE: EU member state grid factor.
    CUSTOM_REGION: User-defined regional boundary.
    """

    COUNTRY = "country"
    EGRID_SUBREGION = "egrid_subregion"
    EU_MEMBER_STATE = "eu_member_state"
    CUSTOM_REGION = "custom_region"

class TDLossSource(str, Enum):
    """Sources of transmission and distribution loss factors.

    IEA: International Energy Agency T&D loss data.
    WORLD_BANK: World Bank development indicators.
    EPA_EGRID: US EPA eGRID subregion loss factors.
    NATIONAL_GRID: National grid operator published data.
    CUSTOM: User-defined or custom loss factors.
    """

    IEA = "iea"
    WORLD_BANK = "world_bank"
    EPA_EGRID = "epa_egrid"
    NATIONAL_GRID = "national_grid"
    CUSTOM = "custom"

class SupplierDataSource(str, Enum):
    """Sources of supplier-specific upstream emission data.

    EPD: Environmental Product Declarations (ISO 14025).
    PCF: Product Carbon Footprint per ISO 14067.
    LCA: Full lifecycle assessment from supplier.
    CDP: CDP Supply Chain Program disclosure.
    MIQ_CERTIFICATE: MiQ methane intensity certificate (gas).
    OGMP2: Oil & Gas Methane Partnership 2.0 data.
    PPA: Power Purchase Agreement renewable attributes.
    GREEN_TARIFF: Green tariff / renewable energy certificate.
    DIRECT_MEASUREMENT: Direct supplier measurement data.
    CUSTOM: User-defined or custom supplier data.
    """

    EPD = "epd"
    PCF = "pcf"
    LCA = "lca"
    CDP = "cdp"
    MIQ_CERTIFICATE = "miq_certificate"
    OGMP2 = "ogmp2"
    PPA = "ppa"
    GREEN_TARIFF = "green_tariff"
    DIRECT_MEASUREMENT = "direct_measurement"
    CUSTOM = "custom"

class AllocationMethod(str, Enum):
    """Allocation methods for multi-product supplier data.

    REVENUE: Allocate by revenue share.
    PRODUCTION_VOLUME: Allocate by production volume share.
    ENERGY_CONTENT: Allocate by energy content share.
    MASS: Allocate by mass share.
    ECONOMIC: Allocate by economic value share.
    """

    REVENUE = "revenue"
    PRODUCTION_VOLUME = "production_volume"
    ENERGY_CONTENT = "energy_content"
    MASS = "mass"
    ECONOMIC = "economic"

class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for spend-based calculations.

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

    VERY_HIGH: Score 1 -- verified primary supplier data.
    HIGH: Score 2 -- established databases (DEFRA, IEA).
    MEDIUM: Score 3 -- industry averages.
    LOW: Score 4 -- broad estimates or proxies.
    VERY_LOW: Score 5 -- unverified assumptions.
    """

    VERY_HIGH = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    VERY_LOW = 5

class UncertaintyMethod(str, Enum):
    """Methods for quantifying uncertainty in emission calculations.

    MONTE_CARLO: Monte Carlo simulation (10,000+ iterations).
    ANALYTICAL: Analytical error propagation (root-sum-of-squares).
    IPCC_DEFAULT: Default uncertainty ranges per IPCC guidelines.
    """

    MONTE_CARLO = "monte_carlo"
    ANALYTICAL = "analytical"
    IPCC_DEFAULT = "ipcc_default"

class ComplianceFramework(str, Enum):
    """Regulatory and voluntary reporting frameworks for Category 3.

    GHG_PROTOCOL_SCOPE3: GHG Protocol Scope 3 Standard Chapter 4.
    CSRD_ESRS_E1: EU CSRD ESRS E1 Scope 3 by category.
    CDP: Carbon Disclosure Project climate questionnaire.
    SBTI: Science Based Targets initiative.
    SB_253: California Senate Bill 253.
    GRI_305: Global Reporting Initiative Standard 305.
    ISO_14064: ISO 14064-1:2018 indirect emissions.
    """

    GHG_PROTOCOL_SCOPE3 = "ghg_protocol_scope3"
    CSRD_ESRS_E1 = "csrd_esrs_e1"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI_305 = "gri_305"
    ISO_14064 = "iso_14064"

class ComplianceStatus(str, Enum):
    """Result of a regulatory compliance check.

    COMPLIANT: All requirements met.
    PARTIALLY_COMPLIANT: Some requirements met but gaps remain.
    NON_COMPLIANT: One or more mandatory requirements not met.
    NOT_APPLICABLE: Framework does not apply to this entity.
    """

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"

class PipelineStage(str, Enum):
    """Stages in the Fuel & Energy Activities calculation pipeline.

    Ten sequential stages from validation through sealing.
    """

    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE_3A = "calculate_3a"
    CALCULATE_3B = "calculate_3b"
    CALCULATE_3C = "calculate_3c"
    COMPLIANCE = "compliance"
    AGGREGATE = "aggregate"
    SEAL = "seal"

class BatchStatus(str, Enum):
    """Status of a batch calculation job.

    PENDING: Created but not started.
    RUNNING: Actively processing.
    COMPLETED: All items completed successfully.
    FAILED: One or more items failed.
    CANCELLED: Job was cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

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
    """Greenhouse gases tracked in Scope 3 Category 3 calculations.

    CO2: Carbon dioxide.
    CH4: Methane.
    N2O: Nitrous oxide.
    CO2E: Carbon dioxide equivalent (aggregate metric).
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO2E = "CO2e"

class AccountingMethod(str, Enum):
    """Electricity accounting methods for Activity 3b and 3c.

    LOCATION_BASED: Uses average grid emission factors for the
        geographic location of electricity consumption.
    MARKET_BASED: Uses supplier-specific, contractual, or
        residual-mix emission factors reflecting procurement
        decisions (PPAs, RECs, green tariffs).
    """

    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"

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
# 2. WTT fuel emission factors (kgCO2e per kWh of fuel energy content)
#    Source: DEFRA 2024, GREET 2023, JEC WTW v5
#    Keys: co2, ch4, n2o, total (all Decimal)
# ---------------------------------------------------------------------------

WTT_FUEL_EMISSION_FACTORS: Dict[FuelType, Dict[str, Decimal]] = {
    # Fossil -- gaseous
    FuelType.NATURAL_GAS: {
        "co2": Decimal("0.02100"),
        "ch4": Decimal("0.00350"),
        "n2o": Decimal("0.00010"),
        "total": Decimal("0.02460"),
    },
    FuelType.LPG: {
        "co2": Decimal("0.02800"),
        "ch4": Decimal("0.00180"),
        "n2o": Decimal("0.00010"),
        "total": Decimal("0.02990"),
    },
    FuelType.PROPANE: {
        "co2": Decimal("0.02650"),
        "ch4": Decimal("0.00170"),
        "n2o": Decimal("0.00010"),
        "total": Decimal("0.02830"),
    },
    # Fossil -- liquid
    FuelType.DIESEL: {
        "co2": Decimal("0.04800"),
        "ch4": Decimal("0.00250"),
        "n2o": Decimal("0.00020"),
        "total": Decimal("0.05070"),
    },
    FuelType.PETROL_GASOLINE: {
        "co2": Decimal("0.04600"),
        "ch4": Decimal("0.00280"),
        "n2o": Decimal("0.00020"),
        "total": Decimal("0.04900"),
    },
    FuelType.FUEL_OIL_2: {
        "co2": Decimal("0.04200"),
        "ch4": Decimal("0.00200"),
        "n2o": Decimal("0.00015"),
        "total": Decimal("0.04415"),
    },
    FuelType.FUEL_OIL_6: {
        "co2": Decimal("0.04500"),
        "ch4": Decimal("0.00220"),
        "n2o": Decimal("0.00018"),
        "total": Decimal("0.04738"),
    },
    FuelType.KEROSENE: {
        "co2": Decimal("0.04100"),
        "ch4": Decimal("0.00190"),
        "n2o": Decimal("0.00012"),
        "total": Decimal("0.04302"),
    },
    FuelType.JET_FUEL: {
        "co2": Decimal("0.04350"),
        "ch4": Decimal("0.00210"),
        "n2o": Decimal("0.00014"),
        "total": Decimal("0.04574"),
    },
    FuelType.PETROLEUM_COKE: {
        "co2": Decimal("0.03800"),
        "ch4": Decimal("0.00150"),
        "n2o": Decimal("0.00008"),
        "total": Decimal("0.03958"),
    },
    FuelType.WASTE_OIL: {
        "co2": Decimal("0.03500"),
        "ch4": Decimal("0.00300"),
        "n2o": Decimal("0.00025"),
        "total": Decimal("0.03825"),
    },
    # Fossil -- solid
    FuelType.COAL_BITUMINOUS: {
        "co2": Decimal("0.03200"),
        "ch4": Decimal("0.00500"),
        "n2o": Decimal("0.00010"),
        "total": Decimal("0.03710"),
    },
    FuelType.COAL_SUB_BITUMINOUS: {
        "co2": Decimal("0.03400"),
        "ch4": Decimal("0.00520"),
        "n2o": Decimal("0.00012"),
        "total": Decimal("0.03932"),
    },
    FuelType.COAL_LIGNITE: {
        "co2": Decimal("0.03600"),
        "ch4": Decimal("0.00550"),
        "n2o": Decimal("0.00015"),
        "total": Decimal("0.04165"),
    },
    FuelType.COAL_ANTHRACITE: {
        "co2": Decimal("0.02900"),
        "ch4": Decimal("0.00400"),
        "n2o": Decimal("0.00008"),
        "total": Decimal("0.03308"),
    },
    FuelType.PEAT: {
        "co2": Decimal("0.03100"),
        "ch4": Decimal("0.00600"),
        "n2o": Decimal("0.00020"),
        "total": Decimal("0.03720"),
    },
    FuelType.MSW: {
        "co2": Decimal("0.01200"),
        "ch4": Decimal("0.00800"),
        "n2o": Decimal("0.00030"),
        "total": Decimal("0.02030"),
    },
    # Biofuels
    FuelType.ETHANOL: {
        "co2": Decimal("0.01500"),
        "ch4": Decimal("0.00120"),
        "n2o": Decimal("0.00250"),
        "total": Decimal("0.01870"),
    },
    FuelType.BIODIESEL: {
        "co2": Decimal("0.01800"),
        "ch4": Decimal("0.00100"),
        "n2o": Decimal("0.00200"),
        "total": Decimal("0.02100"),
    },
    FuelType.BIOGAS: {
        "co2": Decimal("0.00800"),
        "ch4": Decimal("0.00400"),
        "n2o": Decimal("0.00005"),
        "total": Decimal("0.01205"),
    },
    FuelType.HVO: {
        "co2": Decimal("0.01100"),
        "ch4": Decimal("0.00080"),
        "n2o": Decimal("0.00015"),
        "total": Decimal("0.01195"),
    },
    # Biomass
    FuelType.WOOD_PELLETS: {
        "co2": Decimal("0.01400"),
        "ch4": Decimal("0.00150"),
        "n2o": Decimal("0.00018"),
        "total": Decimal("0.01568"),
    },
    FuelType.BIOMASS_SOLID: {
        "co2": Decimal("0.01300"),
        "ch4": Decimal("0.00200"),
        "n2o": Decimal("0.00020"),
        "total": Decimal("0.01520"),
    },
    FuelType.BIOMASS_LIQUID: {
        "co2": Decimal("0.01600"),
        "ch4": Decimal("0.00110"),
        "n2o": Decimal("0.00022"),
        "total": Decimal("0.01732"),
    },
    FuelType.LANDFILL_GAS: {
        "co2": Decimal("0.00500"),
        "ch4": Decimal("0.00600"),
        "n2o": Decimal("0.00003"),
        "total": Decimal("0.01103"),
    },
}

# ---------------------------------------------------------------------------
# 3. Upstream electricity emission factors by country (kgCO2e/kWh)
#    Lifecycle upstream of generation (fuel extraction, processing,
#    transport to power station).
#    Source: IEA 2023, DEFRA 2024, ecoinvent 3.11
# ---------------------------------------------------------------------------

UPSTREAM_ELECTRICITY_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.04500"),
    "GB": Decimal("0.03200"),
    "DE": Decimal("0.05100"),
    "FR": Decimal("0.00800"),
    "JP": Decimal("0.05600"),
    "CN": Decimal("0.07200"),
    "IN": Decimal("0.08100"),
    "AU": Decimal("0.06300"),
    "CA": Decimal("0.01500"),
    "BR": Decimal("0.01200"),
    "KR": Decimal("0.05400"),
    "IT": Decimal("0.03800"),
    "ES": Decimal("0.02900"),
    "NL": Decimal("0.04200"),
    "BE": Decimal("0.03100"),
    "SE": Decimal("0.00600"),
    "NO": Decimal("0.00400"),
    "DK": Decimal("0.02200"),
    "FI": Decimal("0.01800"),
    "AT": Decimal("0.01400"),
    "CH": Decimal("0.00500"),
    "PL": Decimal("0.06800"),
    "CZ": Decimal("0.05500"),
    "PT": Decimal("0.02600"),
    "IE": Decimal("0.03500"),
    "NZ": Decimal("0.01100"),
    "SG": Decimal("0.04800"),
    "ZA": Decimal("0.07500"),
    "MX": Decimal("0.05200"),
    "TH": Decimal("0.05000"),
    "ID": Decimal("0.06500"),
    "MY": Decimal("0.05300"),
    "PH": Decimal("0.05800"),
    "VN": Decimal("0.04900"),
    "AR": Decimal("0.03700"),
    "CL": Decimal("0.03400"),
    "CO": Decimal("0.02100"),
    "PE": Decimal("0.02300"),
    "EG": Decimal("0.04600"),
    "NG": Decimal("0.04400"),
    "KE": Decimal("0.02800"),
    "AE": Decimal("0.05700"),
    "SA": Decimal("0.06100"),
    "TR": Decimal("0.04700"),
    "RU": Decimal("0.05900"),
    "UA": Decimal("0.04300"),
}

# ---------------------------------------------------------------------------
# 4. T&D loss factors by country (fraction, not percentage)
#    Source: IEA 2023, World Bank WDI 2023
# ---------------------------------------------------------------------------

TD_LOSS_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.0500"),
    "GB": Decimal("0.0780"),
    "DE": Decimal("0.0388"),
    "FR": Decimal("0.0620"),
    "JP": Decimal("0.0431"),
    "CN": Decimal("0.0454"),
    "IN": Decimal("0.1900"),
    "AU": Decimal("0.0478"),
    "CA": Decimal("0.0820"),
    "BR": Decimal("0.1577"),
    "KR": Decimal("0.0335"),
    "IT": Decimal("0.0630"),
    "ES": Decimal("0.0890"),
    "NL": Decimal("0.0410"),
    "BE": Decimal("0.0480"),
    "SE": Decimal("0.0660"),
    "NO": Decimal("0.0580"),
    "DK": Decimal("0.0570"),
    "FI": Decimal("0.0350"),
    "AT": Decimal("0.0520"),
    "CH": Decimal("0.0530"),
    "PL": Decimal("0.0690"),
    "CZ": Decimal("0.0560"),
    "PT": Decimal("0.0870"),
    "IE": Decimal("0.0770"),
    "NZ": Decimal("0.0680"),
    "SG": Decimal("0.0220"),
    "ZA": Decimal("0.0890"),
    "MX": Decimal("0.1460"),
    "TH": Decimal("0.0640"),
    "ID": Decimal("0.0950"),
    "MY": Decimal("0.0370"),
    "PH": Decimal("0.0980"),
    "VN": Decimal("0.0730"),
    "AR": Decimal("0.1420"),
    "CL": Decimal("0.0830"),
    "CO": Decimal("0.1250"),
    "PE": Decimal("0.1180"),
    "EG": Decimal("0.1210"),
    "NG": Decimal("0.1650"),
    "KE": Decimal("0.1840"),
    "AE": Decimal("0.0580"),
    "SA": Decimal("0.0720"),
    "TR": Decimal("0.1250"),
    "RU": Decimal("0.1030"),
    "UA": Decimal("0.1180"),
    "PK": Decimal("0.1750"),
    "BD": Decimal("0.1280"),
    "GH": Decimal("0.2100"),
    "TZ": Decimal("0.1920"),
}

# ---------------------------------------------------------------------------
# 5. eGRID subregion T&D loss factors (fraction)
#    Source: EPA eGRID 2022
# ---------------------------------------------------------------------------

EGRID_TD_LOSS_FACTORS: Dict[str, Decimal] = {
    "CAMX": Decimal("0.0518"),
    "ERCT": Decimal("0.0527"),
    "FRCC": Decimal("0.0498"),
    "MROE": Decimal("0.0541"),
    "MROW": Decimal("0.0536"),
    "NEWE": Decimal("0.0487"),
    "NWPP": Decimal("0.0463"),
    "NYCW": Decimal("0.0512"),
    "NYLI": Decimal("0.0534"),
    "NYUP": Decimal("0.0492"),
    "PRMS": Decimal("0.0485"),
    "RFCE": Decimal("0.0508"),
    "RFCM": Decimal("0.0519"),
    "RFCW": Decimal("0.0525"),
    "RMPA": Decimal("0.0549"),
    "SPNO": Decimal("0.0558"),
    "SPSO": Decimal("0.0565"),
    "SRMV": Decimal("0.0543"),
    "SRMW": Decimal("0.0531"),
    "SRSO": Decimal("0.0516"),
    "SRTV": Decimal("0.0523"),
    "SRVC": Decimal("0.0509"),
    "AKGD": Decimal("0.0612"),
    "AKMS": Decimal("0.0598"),
    "HIMS": Decimal("0.0574"),
    "HIOA": Decimal("0.0589"),
}

# ---------------------------------------------------------------------------
# 6. Fuel heating values (NCV in kWh per unit)
#    Keys: unit name -> kWh per that unit
#    Source: IPCC 2006 GL, DEFRA 2024
# ---------------------------------------------------------------------------

FUEL_HEATING_VALUES: Dict[FuelType, Dict[str, Decimal]] = {
    FuelType.NATURAL_GAS: {
        "kwh_per_m3": Decimal("10.55"),
        "kwh_per_therm": Decimal("29.31"),
        "kwh_per_mmbtu": Decimal("293.07"),
    },
    FuelType.DIESEL: {
        "kwh_per_litre": Decimal("10.27"),
        "kwh_per_us_gallon": Decimal("38.87"),
        "kwh_per_kg": Decimal("12.11"),
        "kwh_per_tonne": Decimal("12110"),
    },
    FuelType.PETROL_GASOLINE: {
        "kwh_per_litre": Decimal("9.06"),
        "kwh_per_us_gallon": Decimal("34.30"),
        "kwh_per_kg": Decimal("12.10"),
        "kwh_per_tonne": Decimal("12100"),
    },
    FuelType.COAL_BITUMINOUS: {
        "kwh_per_kg": Decimal("7.50"),
        "kwh_per_tonne": Decimal("7500"),
    },
    FuelType.COAL_SUB_BITUMINOUS: {
        "kwh_per_kg": Decimal("5.28"),
        "kwh_per_tonne": Decimal("5280"),
    },
    FuelType.COAL_LIGNITE: {
        "kwh_per_kg": Decimal("3.89"),
        "kwh_per_tonne": Decimal("3890"),
    },
    FuelType.COAL_ANTHRACITE: {
        "kwh_per_kg": Decimal("8.61"),
        "kwh_per_tonne": Decimal("8610"),
    },
    FuelType.FUEL_OIL_2: {
        "kwh_per_litre": Decimal("10.18"),
        "kwh_per_us_gallon": Decimal("38.54"),
        "kwh_per_kg": Decimal("11.89"),
        "kwh_per_tonne": Decimal("11890"),
    },
    FuelType.FUEL_OIL_6: {
        "kwh_per_litre": Decimal("10.83"),
        "kwh_per_us_gallon": Decimal("41.01"),
        "kwh_per_kg": Decimal("11.28"),
        "kwh_per_tonne": Decimal("11280"),
    },
    FuelType.LPG: {
        "kwh_per_litre": Decimal("7.11"),
        "kwh_per_kg": Decimal("12.78"),
        "kwh_per_tonne": Decimal("12780"),
    },
    FuelType.PROPANE: {
        "kwh_per_litre": Decimal("7.08"),
        "kwh_per_kg": Decimal("13.78"),
        "kwh_per_tonne": Decimal("13780"),
    },
    FuelType.KEROSENE: {
        "kwh_per_litre": Decimal("10.11"),
        "kwh_per_us_gallon": Decimal("38.28"),
        "kwh_per_kg": Decimal("12.53"),
        "kwh_per_tonne": Decimal("12530"),
    },
    FuelType.JET_FUEL: {
        "kwh_per_litre": Decimal("10.00"),
        "kwh_per_us_gallon": Decimal("37.85"),
        "kwh_per_kg": Decimal("12.22"),
        "kwh_per_tonne": Decimal("12220"),
    },
    FuelType.PETROLEUM_COKE: {
        "kwh_per_kg": Decimal("8.89"),
        "kwh_per_tonne": Decimal("8890"),
    },
    FuelType.WASTE_OIL: {
        "kwh_per_litre": Decimal("9.50"),
        "kwh_per_kg": Decimal("10.83"),
        "kwh_per_tonne": Decimal("10830"),
    },
    FuelType.PEAT: {
        "kwh_per_kg": Decimal("2.86"),
        "kwh_per_tonne": Decimal("2860"),
    },
    FuelType.MSW: {
        "kwh_per_kg": Decimal("2.53"),
        "kwh_per_tonne": Decimal("2530"),
    },
    FuelType.ETHANOL: {
        "kwh_per_litre": Decimal("5.87"),
        "kwh_per_kg": Decimal("7.44"),
        "kwh_per_tonne": Decimal("7440"),
    },
    FuelType.BIODIESEL: {
        "kwh_per_litre": Decimal("9.22"),
        "kwh_per_kg": Decimal("10.50"),
        "kwh_per_tonne": Decimal("10500"),
    },
    FuelType.BIOGAS: {
        "kwh_per_m3": Decimal("6.50"),
        "kwh_per_tonne": Decimal("5560"),
    },
    FuelType.HVO: {
        "kwh_per_litre": Decimal("9.44"),
        "kwh_per_kg": Decimal("12.08"),
        "kwh_per_tonne": Decimal("12080"),
    },
    FuelType.WOOD_PELLETS: {
        "kwh_per_kg": Decimal("4.81"),
        "kwh_per_tonne": Decimal("4810"),
    },
    FuelType.BIOMASS_SOLID: {
        "kwh_per_kg": Decimal("4.17"),
        "kwh_per_tonne": Decimal("4170"),
    },
    FuelType.BIOMASS_LIQUID: {
        "kwh_per_litre": Decimal("5.00"),
        "kwh_per_kg": Decimal("5.83"),
        "kwh_per_tonne": Decimal("5830"),
    },
    FuelType.LANDFILL_GAS: {
        "kwh_per_m3": Decimal("5.56"),
        "kwh_per_tonne": Decimal("4750"),
    },
}

# ---------------------------------------------------------------------------
# 7. Fuel density factors (kg per litre) for liquid fuels
#    Source: DEFRA 2024, IPCC 2006 GL
# ---------------------------------------------------------------------------

FUEL_DENSITY_FACTORS: Dict[FuelType, Decimal] = {
    FuelType.DIESEL: Decimal("0.8480"),
    FuelType.PETROL_GASOLINE: Decimal("0.7489"),
    FuelType.FUEL_OIL_2: Decimal("0.8560"),
    FuelType.FUEL_OIL_6: Decimal("0.9600"),
    FuelType.LPG: Decimal("0.5560"),
    FuelType.PROPANE: Decimal("0.5140"),
    FuelType.KEROSENE: Decimal("0.8070"),
    FuelType.JET_FUEL: Decimal("0.8180"),
    FuelType.WASTE_OIL: Decimal("0.8770"),
    FuelType.ETHANOL: Decimal("0.7890"),
    FuelType.BIODIESEL: Decimal("0.8780"),
    FuelType.HVO: Decimal("0.7820"),
    FuelType.BIOMASS_LIQUID: Decimal("0.8570"),
}

# ---------------------------------------------------------------------------
# 8. DQI score numeric values (1=best, 5=worst)
# ---------------------------------------------------------------------------

DQI_SCORE_VALUES: Dict[DQIScore, Decimal] = {
    DQIScore.VERY_HIGH: Decimal("1.0"),
    DQIScore.HIGH: Decimal("2.0"),
    DQIScore.MEDIUM: Decimal("3.0"),
    DQIScore.LOW: Decimal("4.0"),
    DQIScore.VERY_LOW: Decimal("5.0"),
}

# ---------------------------------------------------------------------------
# 9. DQI quality tier labels with composite score ranges
#    (min_score inclusive, max_score exclusive)
# ---------------------------------------------------------------------------

DQI_QUALITY_TIERS: Dict[str, Tuple[Decimal, Decimal]] = {
    "Very High": (Decimal("1.0"), Decimal("1.6")),
    "High": (Decimal("1.6"), Decimal("2.6")),
    "Medium": (Decimal("2.6"), Decimal("3.6")),
    "Low": (Decimal("3.6"), Decimal("4.6")),
    "Very Low": (Decimal("4.6"), Decimal("5.1")),
}

# ---------------------------------------------------------------------------
# 10. Uncertainty ranges by calculation method (min%, max%)
# ---------------------------------------------------------------------------

UNCERTAINTY_RANGES: Dict[CalculationMethod, Tuple[Decimal, Decimal]] = {
    CalculationMethod.SUPPLIER_SPECIFIC: (Decimal("5"), Decimal("10")),
    CalculationMethod.HYBRID: (Decimal("10"), Decimal("25")),
    CalculationMethod.AVERAGE_DATA: (Decimal("15"), Decimal("30")),
    CalculationMethod.SPEND_BASED: (Decimal("30"), Decimal("60")),
}

# ---------------------------------------------------------------------------
# 11. Coverage thresholds by level
# ---------------------------------------------------------------------------

COVERAGE_THRESHOLDS: Dict[str, Decimal] = {
    "full": Decimal("100.0"),
    "high": Decimal("95.0"),
    "medium": Decimal("90.0"),
    "low": Decimal("80.0"),
    "minimal": Decimal("0.0"),
}

# ---------------------------------------------------------------------------
# 12. Emission factor hierarchy priority (1=best, 8=worst)
#     Per GHG Protocol Scope 3 Technical Guidance Section 1.4
# ---------------------------------------------------------------------------

EF_HIERARCHY_PRIORITY: Dict[str, int] = {
    "supplier_miq_verified": 1,
    "supplier_epd_verified": 2,
    "supplier_pcf_verified": 3,
    "supplier_ppa_contract": 4,
    "database_defra_wtt": 5,
    "database_iea_upstream": 6,
    "database_ecoinvent_lca": 7,
    "global_avg_fallback": 8,
}

# ---------------------------------------------------------------------------
# 13. Framework required disclosures for Category 3 compliance
# ---------------------------------------------------------------------------

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[
    ComplianceFramework, List[str]
] = {
    ComplianceFramework.GHG_PROTOCOL_SCOPE3: [
        "category_3_total_tco2e",
        "activity_3a_upstream_fuels_tco2e",
        "activity_3b_upstream_electricity_tco2e",
        "activity_3c_td_losses_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "fuel_consumption_coverage_pct",
        "electricity_consumption_coverage_pct",
        "category_boundary_description",
        "double_counting_prevention_scope1_2",
        "base_year_recalculation_policy",
        "exclusions_and_limitations",
        "gwp_values_used",
        "uncertainty_assessment",
    ],
    ComplianceFramework.CSRD_ESRS_E1: [
        "category_3_total_tco2e",
        "activity_3a_upstream_fuels_tco2e",
        "activity_3b_upstream_electricity_tco2e",
        "activity_3c_td_losses_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "fuel_consumption_coverage_pct",
        "electricity_consumption_coverage_pct",
        "intensity_per_revenue",
        "significant_changes_explanation",
        "base_year_emissions",
        "reduction_targets",
        "value_chain_boundary",
        "gwp_values_used",
        "location_vs_market_based_split",
    ],
    ComplianceFramework.CDP: [
        "category_3_total_tco2e",
        "category_3_relevance",
        "calculation_methodology",
        "percentage_calculated_primary_data",
        "percentage_calculated_secondary_data",
        "emission_factor_sources",
        "fuel_consumption_coverage_pct",
        "electricity_consumption_coverage_pct",
        "verification_status",
        "boundary_description",
        "activity_split_3a_3b_3c",
    ],
    ComplianceFramework.SBTI: [
        "category_3_total_tco2e",
        "base_year_category_3_tco2e",
        "target_year",
        "reduction_percentage",
        "coverage_percentage",
        "supplier_engagement_targets",
        "calculation_methodology",
        "emission_factor_sources",
    ],
    ComplianceFramework.SB_253: [
        "category_3_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "data_quality_assessment",
        "fuel_consumption_coverage_pct",
        "electricity_consumption_coverage_pct",
        "assurance_provider",
        "reporting_entity_revenue",
        "gwp_values_used",
    ],
    ComplianceFramework.GRI_305: [
        "category_3_total_tco2e",
        "calculation_methodology",
        "emission_factor_sources",
        "gwp_values_used",
        "consolidation_approach",
        "base_year_information",
        "standards_and_methodologies",
        "significant_changes",
    ],
    ComplianceFramework.ISO_14064: [
        "category_3_total_tco2e",
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
        "activity_split_3a_3b_3c",
    ],
}

# =============================================================================
# Data Models (25) -- Pydantic v2, frozen=True
# =============================================================================

# ---------------------------------------------------------------------------
# 1. FuelConsumptionRecord
# ---------------------------------------------------------------------------

class FuelConsumptionRecord(GreenLangBase):
    """A single fuel consumption record for Activity 3a WTT calculation.

    Represents one fuel purchase or consumption event from the fuel
    register or ERP system.  Contains the fuel type, quantity,
    unit, reporting period, and optional supplier information for
    supplier-specific WTT factor resolution.

    Attributes:
        record_id: Unique record identifier.
        fuel_type: Type of fuel consumed.
        fuel_category: Broad fuel classification.
        quantity: Quantity of fuel consumed in the given unit.
        unit: Unit of measurement (litre, m3, kg, tonne, kWh,
            therm, MMBtu, US gallon).
        quantity_kwh: Fuel energy content in kWh (computed or
            provided).  If not provided the pipeline normalizes
            from quantity + unit using FUEL_HEATING_VALUES.

from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import ReportFormat
        period_start: Start date of the consumption period.
        period_end: End date of the consumption period.
        reporting_year: Reporting year for emission allocation.
        facility_id: Facility identifier where fuel was consumed.
        facility_name: Human-readable facility name.
        supplier_id: Unique identifier of the fuel supplier.
        supplier_name: Human-readable supplier name.
        country_code: ISO 3166-1 alpha-2 country code.
        is_biogenic: Whether fuel is classified as biogenic.
        metadata: Additional key-value pairs for extensibility.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Type of fuel consumed",
    )
    fuel_category: FuelCategory = Field(
        default=FuelCategory.FOSSIL,
        description="Broad fuel classification",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Quantity of fuel consumed",
    )
    unit: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unit of measurement (litre, m3, kg, tonne, kWh)",
    )
    quantity_kwh: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Fuel energy content in kWh (computed or provided)",
    )
    period_start: date = Field(
        ...,
        description="Start date of the consumption period",
    )
    period_end: date = Field(
        ...,
        description="End date of the consumption period",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year for emission allocation",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Facility identifier where fuel was consumed",
    )
    facility_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable facility name",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Unique identifier of the fuel supplier",
    )
    supplier_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable supplier name",
    )
    country_code: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 country code",
    )
    is_biogenic: bool = Field(
        default=False,
        description="Whether fuel is classified as biogenic",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for extensibility",
    )

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

# ---------------------------------------------------------------------------
# 2. ElectricityConsumptionRecord
# ---------------------------------------------------------------------------

class ElectricityConsumptionRecord(GreenLangBase):
    """A single electricity / energy consumption record for 3b and 3c.

    Represents one electricity (or steam / heating / cooling)
    purchase event.  Contains the energy type, quantity, grid
    region, and accounting method (location vs market based).

    Attributes:
        record_id: Unique record identifier.
        energy_type: Type of energy purchased.
        quantity_kwh: Energy consumed in kWh.
        grid_region: Grid region identifier (country code or
            eGRID subregion).
        grid_region_type: Type of grid region identifier.
        accounting_method: Location-based or market-based.
        period_start: Start date of the consumption period.
        period_end: End date of the consumption period.
        reporting_year: Reporting year for emission allocation.
        facility_id: Facility identifier.
        facility_name: Human-readable facility name.
        supplier_id: Unique identifier of the energy supplier.
        supplier_name: Human-readable supplier name.
        country_code: ISO 3166-1 alpha-2 country code.
        is_renewable: Whether energy is from renewable sources.
        ppa_id: Power Purchase Agreement identifier (if any).
        rec_count: Number of Renewable Energy Certificates.
        metadata: Additional key-value pairs for extensibility.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier",
    )
    energy_type: EnergyType = Field(
        default=EnergyType.ELECTRICITY,
        description="Type of energy purchased",
    )
    quantity_kwh: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Energy consumed in kWh",
    )
    grid_region: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Grid region identifier (country code or eGRID)",
    )
    grid_region_type: GridRegionType = Field(
        default=GridRegionType.COUNTRY,
        description="Type of grid region identifier",
    )
    accounting_method: AccountingMethod = Field(
        default=AccountingMethod.LOCATION_BASED,
        description="Location-based or market-based accounting",
    )
    period_start: date = Field(
        ...,
        description="Start date of the consumption period",
    )
    period_end: date = Field(
        ...,
        description="End date of the consumption period",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year for emission allocation",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Facility identifier",
    )
    facility_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable facility name",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Unique identifier of the energy supplier",
    )
    supplier_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable supplier name",
    )
    country_code: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 country code",
    )
    is_renewable: bool = Field(
        default=False,
        description="Whether energy is from renewable sources",
    )
    ppa_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Power Purchase Agreement identifier",
    )
    rec_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of Renewable Energy Certificates",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for extensibility",
    )

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

# ---------------------------------------------------------------------------
# 3. WTTEmissionFactor
# ---------------------------------------------------------------------------

class WTTEmissionFactor(GreenLangBase):
    """A well-to-tank emission factor for a specific fuel type.

    Represents a single WTT factor entry from a given source,
    year, and region.  Contains per-gas breakdown (CO2, CH4, N2O)
    and total in kgCO2e per kWh of fuel energy content.

    Attributes:
        fuel_type: Fuel type this factor applies to.
        source: Source database or publication.
        co2: CO2 component in kgCO2e per kWh.
        ch4: CH4 component in kgCO2e per kWh.
        n2o: N2O component in kgCO2e per kWh.
        total: Total WTT factor in kgCO2e per kWh.
        unit: Denominator unit (always kgCO2e/kWh).
        year: Reference year of the factor.
        region: Geographic region the factor applies to.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    fuel_type: FuelType = Field(
        ...,
        description="Fuel type this factor applies to",
    )
    source: WTTFactorSource = Field(
        ...,
        description="Source database or publication",
    )
    co2: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2 component in kgCO2e per kWh",
    )
    ch4: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CH4 component in kgCO2e per kWh",
    )
    n2o: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="N2O component in kgCO2e per kWh",
    )
    total: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total WTT factor in kgCO2e per kWh",
    )
    unit: str = Field(
        default="kgCO2e/kWh",
        max_length=50,
        description="Denominator unit",
    )
    year: int = Field(
        default=2024,
        ge=2000,
        le=2100,
        description="Reference year of the factor",
    )
    region: str = Field(
        default="GLOBAL",
        max_length=50,
        description="Geographic region the factor applies to",
    )

# ---------------------------------------------------------------------------
# 4. UpstreamElectricityFactor
# ---------------------------------------------------------------------------

class UpstreamElectricityFactor(GreenLangBase):
    """Upstream lifecycle emission factor for grid electricity.

    Represents the upstream (pre-generation) emission factor for
    electricity in a specific country / region.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        upstream_ef: Upstream factor in kgCO2e per kWh.
        source: Source of the factor.
        year: Reference year of the factor.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 country code",
    )
    upstream_ef: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Upstream emission factor in kgCO2e per kWh",
    )
    source: str = Field(
        default="IEA",
        max_length=100,
        description="Source of the factor",
    )
    year: int = Field(
        default=2023,
        ge=2000,
        le=2100,
        description="Reference year of the factor",
    )

# ---------------------------------------------------------------------------
# 5. TDLossFactor
# ---------------------------------------------------------------------------

class TDLossFactor(GreenLangBase):
    """Transmission and distribution loss factor for a region.

    Represents the percentage of electricity lost during
    transmission and distribution for a specific country or grid
    subregion.

    Attributes:
        country_code: ISO country code or eGRID subregion code.
        loss_percentage: T&D loss as a fraction (e.g. 0.05 = 5%).
        source: Source of the loss factor data.
        year: Reference year of the factor.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="ISO country code or eGRID subregion code",
    )
    loss_percentage: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="T&D loss as a fraction (e.g. 0.05 = 5%)",
    )
    source: TDLossSource = Field(
        default=TDLossSource.IEA,
        description="Source of the loss factor data",
    )
    year: int = Field(
        default=2023,
        ge=2000,
        le=2100,
        description="Reference year of the factor",
    )

# ---------------------------------------------------------------------------
# 6. SupplierFuelData
# ---------------------------------------------------------------------------

class SupplierFuelData(GreenLangBase):
    """Supplier-specific upstream emission data for a fuel type.

    Contains primary data from a fuel supplier including upstream
    emission factors, verification level, and certificate or EPD
    references.

    Attributes:
        supplier_id: Unique supplier identifier.
        supplier_name: Human-readable supplier name.
        fuel_type: Fuel type the data applies to.
        upstream_ef: Supplier-specific WTT EF (kgCO2e/kWh).
        verification_level: Level of third-party verification.
        epd_number: EPD registration number (if applicable).
        miq_grade: MiQ methane intensity grade (A-F).
        ogmp2_level: OGMP2 reporting level (1-5).
        data_source: Source type of the supplier data.
        reporting_year: Year the supplier data was reported.
        allocation_method: Allocation method if multi-product.
        allocation_factor: Allocation factor applied (0-1).
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique supplier identifier",
    )
    supplier_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable supplier name",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Fuel type the data applies to",
    )
    upstream_ef: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Supplier-specific WTT EF in kgCO2e per kWh",
    )
    verification_level: str = Field(
        default="unverified",
        max_length=50,
        description="Level of third-party verification",
    )
    epd_number: Optional[str] = Field(
        default=None,
        max_length=100,
        description="EPD registration number if applicable",
    )
    miq_grade: Optional[str] = Field(
        default=None,
        max_length=5,
        description="MiQ methane intensity grade (A-F)",
    )
    ogmp2_level: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="OGMP2 reporting level (1-5)",
    )
    data_source: SupplierDataSource = Field(
        default=SupplierDataSource.DIRECT_MEASUREMENT,
        description="Source type of the supplier data",
    )
    reporting_year: Optional[int] = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Year the supplier data was reported",
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.ENERGY_CONTENT,
        description="Allocation method if multi-product supplier",
    )
    allocation_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Allocation factor applied (0-1)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

# ---------------------------------------------------------------------------
# 7. Activity3aResult
# ---------------------------------------------------------------------------

class Activity3aResult(GreenLangBase):
    """Result of Activity 3a (upstream fuel emissions) calculation.

    Contains the WTT emissions for one fuel consumption record,
    broken down by gas (CO2, CH4, N2O) and total CO2e.

    Attributes:
        record_id: Unique result identifier.
        fuel_record_id: Reference to the source fuel record.
        fuel_type: Fuel type processed.
        fuel_category: Broad fuel classification.
        fuel_consumed_kwh: Fuel energy content in kWh.
        wtt_ef_total: WTT emission factor used (kgCO2e/kWh).
        wtt_ef_source: Source of the WTT factor.
        emissions_co2: CO2 component in kgCO2e.
        emissions_ch4: CH4 component in kgCO2e.
        emissions_n2o: N2O component in kgCO2e.
        emissions_total: Total WTT emissions in kgCO2e.
        is_biogenic: Whether emissions are classified as biogenic.
        dqi_score: Composite data quality score (1-5).
        uncertainty_pct: Uncertainty percentage (+/-).
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier",
    )
    fuel_record_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source fuel record",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Fuel type processed",
    )
    fuel_category: FuelCategory = Field(
        default=FuelCategory.FOSSIL,
        description="Broad fuel classification",
    )
    fuel_consumed_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Fuel energy content in kWh",
    )
    wtt_ef_total: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="WTT emission factor used (kgCO2e/kWh)",
    )
    wtt_ef_source: str = Field(
        default="DEFRA",
        max_length=100,
        description="Source of the WTT factor",
    )
    emissions_co2: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CO2 component in kgCO2e",
    )
    emissions_ch4: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 component in kgCO2e",
    )
    emissions_n2o: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="N2O component in kgCO2e",
    )
    emissions_total: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total WTT emissions in kgCO2e",
    )
    is_biogenic: bool = Field(
        default=False,
        description="Whether emissions are classified as biogenic",
    )
    dqi_score: Decimal = Field(
        default=Decimal("3.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Composite data quality score (1-5)",
    )
    uncertainty_pct: Decimal = Field(
        default=Decimal("25.0"),
        ge=Decimal("0"),
        description="Uncertainty percentage (+/-)",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash for audit trail",
    )

# ---------------------------------------------------------------------------
# 8. Activity3bResult
# ---------------------------------------------------------------------------

class Activity3bResult(GreenLangBase):
    """Result of Activity 3b (upstream electricity emissions) calculation.

    Contains the upstream lifecycle emissions for one electricity
    consumption record including the accounting method used.

    Attributes:
        record_id: Unique result identifier.
        electricity_record_id: Reference to the source record.
        energy_type: Type of energy (electricity, steam, etc.).
        energy_consumed_kwh: Energy consumed in kWh.
        upstream_ef: Upstream emission factor used (kgCO2e/kWh).
        upstream_ef_source: Source of the upstream factor.
        accounting_method: Location-based or market-based.
        grid_region: Grid region used for factor lookup.
        emissions_total: Total upstream emissions in kgCO2e.
        is_renewable: Whether from renewable source.
        dqi_score: Composite data quality score (1-5).
        uncertainty_pct: Uncertainty percentage (+/-).
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier",
    )
    electricity_record_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source electricity record",
    )
    energy_type: EnergyType = Field(
        default=EnergyType.ELECTRICITY,
        description="Type of energy",
    )
    energy_consumed_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Energy consumed in kWh",
    )
    upstream_ef: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Upstream emission factor in kgCO2e per kWh",
    )
    upstream_ef_source: str = Field(
        default="IEA",
        max_length=100,
        description="Source of the upstream factor",
    )
    accounting_method: AccountingMethod = Field(
        default=AccountingMethod.LOCATION_BASED,
        description="Location-based or market-based accounting",
    )
    grid_region: str = Field(
        default="",
        max_length=50,
        description="Grid region used for factor lookup",
    )
    emissions_total: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total upstream emissions in kgCO2e",
    )
    is_renewable: bool = Field(
        default=False,
        description="Whether from renewable source",
    )
    dqi_score: Decimal = Field(
        default=Decimal("3.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Composite data quality score (1-5)",
    )
    uncertainty_pct: Decimal = Field(
        default=Decimal("25.0"),
        ge=Decimal("0"),
        description="Uncertainty percentage (+/-)",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash for audit trail",
    )

# ---------------------------------------------------------------------------
# 9. Activity3cResult
# ---------------------------------------------------------------------------

class Activity3cResult(GreenLangBase):
    """Result of Activity 3c (T&D losses) calculation.

    Contains the emissions attributable to transmission and
    distribution losses for one electricity consumption record,
    including both generation-phase and upstream-phase losses.

    Formula:
        generation_losses = electricity_consumed * td_loss_pct * grid_ef
        upstream_losses   = electricity_consumed * td_loss_pct * upstream_ef
        emissions_total   = generation_losses + upstream_losses

    Attributes:
        record_id: Unique result identifier.
        electricity_record_id: Reference to the source record.
        electricity_consumed_kwh: Electricity consumed in kWh.
        td_loss_pct: T&D loss fraction applied.
        td_loss_source: Source of the T&D loss factor.
        grid_ef: Grid generation emission factor (kgCO2e/kWh).
        upstream_ef: Upstream emission factor (kgCO2e/kWh).
        generation_losses: Generation-phase T&D loss emissions.
        upstream_losses: Upstream-phase T&D loss emissions.
        emissions_total: Total T&D loss emissions in kgCO2e.
        grid_region: Grid region used for factor lookup.
        accounting_method: Location-based or market-based.
        dqi_score: Composite data quality score (1-5).
        uncertainty_pct: Uncertainty percentage (+/-).
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier",
    )
    electricity_record_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source electricity record",
    )
    electricity_consumed_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Electricity consumed in kWh",
    )
    td_loss_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="T&D loss fraction applied",
    )
    td_loss_source: str = Field(
        default="IEA",
        max_length=100,
        description="Source of the T&D loss factor",
    )
    grid_ef: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Grid generation emission factor (kgCO2e/kWh)",
    )
    upstream_ef: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Upstream emission factor (kgCO2e/kWh)",
    )
    generation_losses: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Generation-phase T&D loss emissions in kgCO2e",
    )
    upstream_losses: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Upstream-phase T&D loss emissions in kgCO2e",
    )
    emissions_total: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total T&D loss emissions in kgCO2e",
    )
    grid_region: str = Field(
        default="",
        max_length=50,
        description="Grid region used for factor lookup",
    )
    accounting_method: AccountingMethod = Field(
        default=AccountingMethod.LOCATION_BASED,
        description="Location-based or market-based accounting",
    )
    dqi_score: Decimal = Field(
        default=Decimal("3.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Composite data quality score (1-5)",
    )
    uncertainty_pct: Decimal = Field(
        default=Decimal("30.0"),
        ge=Decimal("0"),
        description="Uncertainty percentage (+/-)",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash for audit trail",
    )

# ---------------------------------------------------------------------------
# 10. Activity3dResult
# ---------------------------------------------------------------------------

class Activity3dResult(GreenLangBase):
    """Result of Activity 3d (generation of electricity sold) calculation.

    Applicable to utilities and energy resellers only.  Contains
    the full lifecycle emissions of electricity generated by the
    reporting entity and sold to end users.

    Attributes:
        record_id: Unique result identifier.
        electricity_sold_kwh: Electricity sold in kWh.
        lifecycle_ef: Full lifecycle EF (kgCO2e/kWh).
        lifecycle_ef_source: Source of the lifecycle factor.
        emissions_total: Total lifecycle emissions in kgCO2e.
        generation_type: Type of generation (fossil, renewable).
        dqi_score: Composite data quality score (1-5).
        uncertainty_pct: Uncertainty percentage (+/-).
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier",
    )
    electricity_sold_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Electricity sold to end users in kWh",
    )
    lifecycle_ef: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Full lifecycle emission factor (kgCO2e/kWh)",
    )
    lifecycle_ef_source: str = Field(
        default="IEA",
        max_length=100,
        description="Source of the lifecycle factor",
    )
    emissions_total: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total lifecycle emissions in kgCO2e",
    )
    generation_type: str = Field(
        default="grid_mix",
        max_length=100,
        description="Type of generation (fossil, renewable, grid_mix)",
    )
    dqi_score: Decimal = Field(
        default=Decimal("3.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Composite data quality score (1-5)",
    )
    uncertainty_pct: Decimal = Field(
        default=Decimal("30.0"),
        ge=Decimal("0"),
        description="Uncertainty percentage (+/-)",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash for audit trail",
    )

# ---------------------------------------------------------------------------
# 11. CalculationResult
# ---------------------------------------------------------------------------

class CalculationResult(GreenLangBase):
    """Complete output of a Category 3 emission calculation run.

    The primary output of the calculation pipeline, containing
    all activity results (3a/3b/3c/3d), total emissions, the
    calculation method used, GWP source, and provenance hash.

    Attributes:
        calc_id: Unique calculation identifier.
        tenant_id: Tenant identifier for multi-tenancy.
        activity_3a_results: List of Activity 3a line-item results.
        activity_3b_results: List of Activity 3b line-item results.
        activity_3c_results: List of Activity 3c line-item results.
        activity_3d_results: List of Activity 3d line-item results.
        total_emissions_kg_co2e: Grand total in kgCO2e.
        total_emissions_tco2e: Grand total in tCO2e.
        total_3a_kg_co2e: Activity 3a subtotal in kgCO2e.
        total_3b_kg_co2e: Activity 3b subtotal in kgCO2e.
        total_3c_kg_co2e: Activity 3c subtotal in kgCO2e.
        total_3d_kg_co2e: Activity 3d subtotal in kgCO2e.
        method: Calculation method used.
        gwp_source: IPCC AR version for GWP values.
        reporting_year: Reporting year.
        provenance_hash: SHA-256 hash over entire result.
        timestamp: UTC timestamp of calculation completion.
        processing_time_ms: Processing duration in ms.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    calc_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation identifier",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy",
    )
    activity_3a_results: List[Activity3aResult] = Field(
        default_factory=list,
        description="Activity 3a (upstream fuels) line-item results",
    )
    activity_3b_results: List[Activity3bResult] = Field(
        default_factory=list,
        description="Activity 3b (upstream electricity) line-item results",
    )
    activity_3c_results: List[Activity3cResult] = Field(
        default_factory=list,
        description="Activity 3c (T&D losses) line-item results",
    )
    activity_3d_results: List[Activity3dResult] = Field(
        default_factory=list,
        description="Activity 3d (electricity sold) line-item results",
    )
    total_emissions_kg_co2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Grand total Category 3 emissions in kgCO2e",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Grand total Category 3 emissions in tCO2e",
    )
    total_3a_kg_co2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Activity 3a subtotal in kgCO2e",
    )
    total_3b_kg_co2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Activity 3b subtotal in kgCO2e",
    )
    total_3c_kg_co2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Activity 3c subtotal in kgCO2e",
    )
    total_3d_kg_co2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Activity 3d subtotal in kgCO2e",
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.AVERAGE_DATA,
        description="Calculation method used",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR5,
        description="IPCC AR version for GWP values",
    )
    reporting_year: int = Field(
        default=2024,
        ge=2000,
        le=2100,
        description="Reporting year",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash over entire result",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of calculation completion",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )

# ---------------------------------------------------------------------------
# 12. GasBreakdown
# ---------------------------------------------------------------------------

class GasBreakdown(GreenLangBase):
    """Per-gas emission breakdown for a calculation result.

    Provides individual gas values (CO2, CH4, N2O) and the
    aggregated CO2e total using a specified GWP source.

    Attributes:
        co2: CO2 emissions in kgCO2.
        ch4: CH4 emissions in kgCH4.
        n2o: N2O emissions in kgN2O.
        co2e: Total in kgCO2e using the specified GWP.
        gwp_source: IPCC AR version used for conversion.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    co2: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CO2 emissions in kgCO2",
    )
    ch4: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 emissions in kgCH4",
    )
    n2o: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="N2O emissions in kgN2O",
    )
    co2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total emissions in kgCO2e",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR5,
        description="IPCC AR version used for GWP conversion",
    )

# ---------------------------------------------------------------------------
# 13. DQIAssessment
# ---------------------------------------------------------------------------

class DQIAssessment(GreenLangBase):
    """Data quality indicator assessment for a calculation result.

    Scores data quality across the five GHG Protocol dimensions
    and computes a composite score (arithmetic mean).  Lower
    scores indicate higher quality.

    Attributes:
        record_id: Reference to the source record.
        activity_type: Category 3 sub-activity assessed.
        temporal: Temporal representativeness score (1-5).
        geographical: Geographical representativeness score (1-5).
        technological: Technological representativeness score (1-5).
        completeness: Data completeness score (1-5).
        reliability: Data reliability score (1-5).
        composite: Arithmetic mean of all five scores.
        tier: Qualitative quality tier label.
        findings: List of findings and recommendations.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the source record",
    )
    activity_type: ActivityType = Field(
        ...,
        description="Category 3 sub-activity assessed",
    )
    temporal: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Temporal representativeness score (1-5)",
    )
    geographical: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Geographical representativeness score (1-5)",
    )
    technological: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Technological representativeness score (1-5)",
    )
    completeness: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Data completeness score (1-5)",
    )
    reliability: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Data reliability score (1-5)",
    )
    composite: Decimal = Field(
        ...,
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
        description="Arithmetic mean of all five scores",
    )
    tier: str = Field(
        default="",
        max_length=50,
        description="Qualitative quality tier label",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="List of findings and recommendations",
    )

# ---------------------------------------------------------------------------
# 14. UncertaintyResult
# ---------------------------------------------------------------------------

class UncertaintyResult(GreenLangBase):
    """Uncertainty quantification result for a calculation.

    Provides statistical metrics for the uncertainty of the
    emission calculation including confidence interval bounds.

    Attributes:
        mean: Mean emission value in kgCO2e.
        std_dev: Standard deviation in kgCO2e.
        cv: Coefficient of variation (std_dev / mean).
        ci_lower: Lower bound of confidence interval (kgCO2e).
        ci_upper: Upper bound of confidence interval (kgCO2e).
        confidence_level: Confidence level percentage (e.g. 95).
        method: Uncertainty quantification method used.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    mean: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Mean emission value in kgCO2e",
    )
    std_dev: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Standard deviation in kgCO2e",
    )
    cv: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Coefficient of variation (std_dev / mean)",
    )
    ci_lower: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Lower bound of confidence interval (kgCO2e)",
    )
    ci_upper: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Upper bound of confidence interval (kgCO2e)",
    )
    confidence_level: Decimal = Field(
        default=Decimal("95.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Confidence level percentage",
    )
    method: UncertaintyMethod = Field(
        default=UncertaintyMethod.IPCC_DEFAULT,
        description="Uncertainty quantification method used",
    )

# ---------------------------------------------------------------------------
# 15. ComplianceCheckResult
# ---------------------------------------------------------------------------

class ComplianceCheckResult(GreenLangBase):
    """Result of a compliance check against one regulatory framework.

    Aggregates individual compliance findings into an overall
    compliance status for a specific framework.

    Attributes:
        framework: The regulatory framework checked.
        status: Overall compliance status.
        findings: List of individual compliance findings.
        score: Compliance score (0-100).
        checked_at: UTC timestamp of the compliance check.
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
    findings: List[ComplianceFinding] = Field(
        default_factory=list,
        description="List of individual compliance findings",
    )
    score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Compliance score (0-100)",
    )
    checked_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the compliance check",
    )

# ---------------------------------------------------------------------------
# 16. ComplianceFinding
# ---------------------------------------------------------------------------

class ComplianceFinding(GreenLangBase):
    """A single compliance finding within a framework check.

    Represents one disclosure or data requirement that was
    evaluated, including its status, severity, and actionable
    recommendation.

    Attributes:
        rule_id: Machine-readable rule identifier.
        rule_name: Human-readable rule name.
        status: Compliance status for this specific rule.
        severity: Severity level (critical, major, minor, info).
        message: Detailed finding message.
        recommendation: Actionable recommendation to remediate.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    rule_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Machine-readable rule identifier",
    )
    rule_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable rule name",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Compliance status for this specific rule",
    )
    severity: str = Field(
        default="info",
        max_length=20,
        description="Severity level (critical, major, minor, info)",
    )
    message: str = Field(
        default="",
        max_length=2000,
        description="Detailed finding message",
    )
    recommendation: str = Field(
        default="",
        max_length=2000,
        description="Actionable recommendation to remediate",
    )

# ---------------------------------------------------------------------------
# 17. PipelineResult
# ---------------------------------------------------------------------------

class PipelineResult(GreenLangBase):
    """Complete output of the Fuel & Energy Activities pipeline.

    Wraps the calculation result with pipeline-level metadata
    including stages completed, compliance results, DQI, and
    provenance.

    Attributes:
        pipeline_id: Unique pipeline execution identifier.
        tenant_id: Tenant identifier for multi-tenancy.
        stages_completed: List of completed pipeline stage names.
        activity_3a_results: Activity 3a line-item results.
        activity_3b_results: Activity 3b line-item results.
        activity_3c_results: Activity 3c line-item results.
        activity_3d_results: Activity 3d line-item results.
        total_emissions_kg_co2e: Grand total in kgCO2e.
        total_emissions_tco2e: Grand total in tCO2e.
        gas_breakdown: Per-gas emission breakdown.
        compliance_results: Compliance check per framework.
        dqi: Overall data quality assessment.
        uncertainty: Uncertainty quantification.
        provenance_hash: SHA-256 hash over entire pipeline output.
        timestamp: UTC timestamp of pipeline completion.
        processing_time_ms: Total processing duration in ms.
        warnings: Warning messages generated during pipeline.
        errors: Error messages generated during pipeline.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique pipeline execution identifier",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy",
    )
    stages_completed: List[str] = Field(
        default_factory=list,
        description="List of completed pipeline stage names",
    )
    activity_3a_results: List[Activity3aResult] = Field(
        default_factory=list,
        description="Activity 3a (upstream fuels) line-item results",
    )
    activity_3b_results: List[Activity3bResult] = Field(
        default_factory=list,
        description="Activity 3b (upstream electricity) line-item results",
    )
    activity_3c_results: List[Activity3cResult] = Field(
        default_factory=list,
        description="Activity 3c (T&D losses) line-item results",
    )
    activity_3d_results: List[Activity3dResult] = Field(
        default_factory=list,
        description="Activity 3d (electricity sold) line-item results",
    )
    total_emissions_kg_co2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Grand total Category 3 emissions in kgCO2e",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Grand total Category 3 emissions in tCO2e",
    )
    gas_breakdown: Optional[GasBreakdown] = Field(
        default=None,
        description="Per-gas emission breakdown",
    )
    compliance_results: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="Compliance check results per framework",
    )
    dqi: Optional[DQIAssessment] = Field(
        default=None,
        description="Overall data quality assessment",
    )
    uncertainty: Optional[UncertaintyResult] = Field(
        default=None,
        description="Uncertainty quantification result",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash over entire pipeline output",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of pipeline completion",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages generated during pipeline",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages generated during pipeline",
    )

# ---------------------------------------------------------------------------
# 18. BatchRequest
# ---------------------------------------------------------------------------

class BatchRequest(GreenLangBase):
    """Batch request for processing multiple fuel and electricity records.

    Enables batch processing of Category 3 calculations for
    multiple records in a single request.

    Attributes:
        batch_id: Unique batch job identifier.
        tenant_id: Tenant identifier for multi-tenancy.
        fuel_records: List of fuel consumption records (3a).
        electricity_records: List of electricity records (3b/3c).
        method: Preferred calculation method.
        gwp_source: IPCC AR version for GWP values.
        compliance_frameworks: Frameworks to check against.
        include_3d: Whether to include Activity 3d (utilities).
        reporting_year: Reporting year for emission allocation.
        metadata: Additional key-value pairs.
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
    fuel_records: List[FuelConsumptionRecord] = Field(
        default_factory=list,
        description="List of fuel consumption records for Activity 3a",
    )
    electricity_records: List[ElectricityConsumptionRecord] = Field(
        default_factory=list,
        description="List of electricity records for Activity 3b/3c",
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.AVERAGE_DATA,
        description="Preferred calculation method",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR5,
        description="IPCC AR version for GWP values",
    )
    compliance_frameworks: Optional[List[ComplianceFramework]] = Field(
        default=None,
        description="Frameworks to check compliance against",
    )
    include_3d: bool = Field(
        default=False,
        description="Whether to include Activity 3d (utilities only)",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year for emission allocation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

    @field_validator("fuel_records")
    @classmethod
    def _validate_fuel_records_count(
        cls, v: List[FuelConsumptionRecord]
    ) -> List[FuelConsumptionRecord]:
        """Validate that fuel records do not exceed maximum."""
        if len(v) > MAX_FUEL_RECORDS:
            raise ValueError(
                f"Maximum {MAX_FUEL_RECORDS} fuel records per "
                f"batch, got {len(v)}"
            )
        return v

    @field_validator("electricity_records")
    @classmethod
    def _validate_electricity_records_count(
        cls, v: List[ElectricityConsumptionRecord]
    ) -> List[ElectricityConsumptionRecord]:
        """Validate that electricity records do not exceed maximum."""
        if len(v) > MAX_ELECTRICITY_RECORDS:
            raise ValueError(
                f"Maximum {MAX_ELECTRICITY_RECORDS} electricity "
                f"records per batch, got {len(v)}"
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
# 19. BatchResult
# ---------------------------------------------------------------------------

class BatchResult(GreenLangBase):
    """Result of a batch calculation job.

    Contains the aggregated results of all records in the batch,
    a summary breakdown by activity type, and job status.

    Attributes:
        batch_id: Reference to the batch request.
        tenant_id: Tenant identifier.
        results: List of individual calculation results.
        summary: Summary breakdown by activity type (tCO2e).
        total_emissions_tco2e: Grand total emissions in tCO2e.
        records_processed: Number of records processed.
        records_failed: Number of records that failed.
        status: Overall batch job status.
        timestamp: UTC timestamp of batch completion.
        processing_time_ms: Total processing duration in ms.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    batch_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the batch request",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    summary: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Summary breakdown by activity type (tCO2e)",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Grand total emissions in tCO2e",
    )
    records_processed: int = Field(
        default=0,
        ge=0,
        description="Number of records processed",
    )
    records_failed: int = Field(
        default=0,
        ge=0,
        description="Number of records that failed",
    )
    status: BatchStatus = Field(
        default=BatchStatus.PENDING,
        description="Overall batch job status",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of batch completion",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )

# ---------------------------------------------------------------------------
# 20. AggregationResult
# ---------------------------------------------------------------------------

class AggregationResult(GreenLangBase):
    """Multi-dimension aggregation of Category 3 results.

    Provides breakdowns by activity type, fuel type, grid region,
    facility, supplier, and reporting period.

    Attributes:
        aggregation_id: Unique aggregation identifier.
        dimension: Primary aggregation dimension name.
        groups: Aggregated emissions by group key (tCO2e).
        total_emissions_tco2e: Total aggregated emissions.
        by_activity: Emissions by activity type (tCO2e).
        by_fuel_type: Emissions by fuel type (tCO2e).
        by_grid_region: Emissions by grid region (tCO2e).
        by_facility: Emissions by facility (tCO2e).
        by_supplier: Emissions by supplier (tCO2e).
        period: Reporting period descriptor.
        provenance_hash: SHA-256 hash of the aggregation.
        timestamp: UTC timestamp of aggregation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    aggregation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique aggregation identifier",
    )
    dimension: str = Field(
        default="activity_type",
        max_length=100,
        description="Primary aggregation dimension name",
    )
    groups: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aggregated emissions by group key (tCO2e)",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total aggregated emissions in tCO2e",
    )
    by_activity: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by activity type (tCO2e)",
    )
    by_fuel_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by fuel type (tCO2e)",
    )
    by_grid_region: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by grid region (tCO2e)",
    )
    by_facility: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by facility (tCO2e)",
    )
    by_supplier: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by supplier (tCO2e)",
    )
    period: str = Field(
        default="",
        max_length=50,
        description="Reporting period descriptor (e.g. 2024)",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the aggregation",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of aggregation",
    )

# ---------------------------------------------------------------------------
# 21. ExportRequest
# ---------------------------------------------------------------------------

class ExportRequest(GreenLangBase):
    """Request to export calculation results in a specific format.

    Attributes:
        export_id: Unique export request identifier.
        format: Requested export format.
        calculation_ids: List of calculation IDs to export.
        include_details: Whether to include line-item details.
        include_compliance: Whether to include compliance results.
        include_dqi: Whether to include DQI assessments.
        include_uncertainty: Whether to include uncertainty data.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    export_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique export request identifier",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Requested export format",
    )
    calculation_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of calculation IDs to export",
    )
    include_details: bool = Field(
        default=True,
        description="Whether to include line-item details",
    )
    include_compliance: bool = Field(
        default=True,
        description="Whether to include compliance results",
    )
    include_dqi: bool = Field(
        default=True,
        description="Whether to include DQI assessments",
    )
    include_uncertainty: bool = Field(
        default=True,
        description="Whether to include uncertainty data",
    )

# ---------------------------------------------------------------------------
# 22. MaterialityResult
# ---------------------------------------------------------------------------

class MaterialityResult(GreenLangBase):
    """Materiality assessment of Category 3 relative to total emissions.

    Evaluates the significance of Category 3 emissions compared
    to Scope 1, Scope 2, and total Scope 3 emissions.

    Attributes:
        total_cat3_tco2e: Total Category 3 emissions in tCO2e.
        scope1_total_tco2e: Total Scope 1 emissions in tCO2e.
        scope2_total_tco2e: Total Scope 2 emissions in tCO2e.
        cat3_pct_of_total: Category 3 as % of total emissions.
        cat3_pct_of_scope3: Category 3 as % of total Scope 3.
        by_activity: Breakdown by activity type (tCO2e).
        is_material: Whether Category 3 is material (> 1%).
        materiality_threshold_pct: Threshold used for assessment.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    total_cat3_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total Category 3 emissions in tCO2e",
    )
    scope1_total_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total Scope 1 emissions in tCO2e",
    )
    scope2_total_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total Scope 2 emissions in tCO2e",
    )
    cat3_pct_of_total: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Category 3 as percentage of total emissions",
    )
    cat3_pct_of_scope3: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Category 3 as percentage of total Scope 3",
    )
    by_activity: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Breakdown by activity type (tCO2e)",
    )
    is_material: bool = Field(
        default=False,
        description="Whether Category 3 is material (above threshold)",
    )
    materiality_threshold_pct: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Materiality threshold percentage used",
    )

# ---------------------------------------------------------------------------
# 23. HotSpotResult
# ---------------------------------------------------------------------------

class HotSpotResult(GreenLangBase):
    """A single hot-spot entry in the Pareto analysis.

    Represents one fuel type, grid region, or supplier ranked
    by emission contribution for 80/20 Pareto analysis.

    Attributes:
        identifier: Fuel type, region, or supplier name.
        identifier_type: Type of identifier (fuel_type, region,
            supplier, facility).
        activity_type: Category 3 sub-activity.
        emissions_tco2e: Emissions in tCO2e.
        pct_of_total: Percentage of total Category 3 emissions.
        rank: Rank in the Pareto ordering (1 = highest).
        cumulative_pct: Cumulative percentage for Pareto curve.
        is_pareto_80: Whether within the top 80% cumulative.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    identifier: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Fuel type, region, or supplier name",
    )
    identifier_type: str = Field(
        default="fuel_type",
        max_length=50,
        description="Type of identifier (fuel_type, region, supplier)",
    )
    activity_type: ActivityType = Field(
        ...,
        description="Category 3 sub-activity",
    )
    emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emissions in tCO2e",
    )
    pct_of_total: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of total Category 3 emissions",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Rank in Pareto ordering (1 = highest)",
    )
    cumulative_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Cumulative percentage for Pareto curve",
    )
    is_pareto_80: bool = Field(
        default=False,
        description="Whether within the top 80% cumulative",
    )

# ---------------------------------------------------------------------------
# 24. YoYDecomposition
# ---------------------------------------------------------------------------

class YoYDecomposition(GreenLangBase):
    """Year-over-year decomposition of Category 3 emission changes.

    Decomposes the change in Category 3 emissions between a base
    year and current year into three drivers: activity change
    (fuel/electricity volume), emission factor change (WTT/grid
    EF updates), and fuel/energy mix change.

    Formula:
        total_change = activity_change + ef_change + mix_change

    Attributes:
        base_year: Base year for comparison.
        current_year: Current reporting year.
        base_year_emissions_tco2e: Base year total in tCO2e.
        current_year_emissions_tco2e: Current year total in tCO2e.
        activity_change_tco2e: Change from volume differences.
        ef_change_tco2e: Change from emission factor updates.
        mix_change_tco2e: Change from fuel/energy mix shifts.
        total_change_tco2e: Total change in tCO2e.
        total_change_pct: Total change as percentage.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    base_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Base year for comparison",
    )
    current_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Current reporting year",
    )
    base_year_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Base year total emissions in tCO2e",
    )
    current_year_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Current year total emissions in tCO2e",
    )
    activity_change_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Change from volume differences in tCO2e",
    )
    ef_change_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Change from emission factor updates in tCO2e",
    )
    mix_change_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Change from fuel/energy mix shifts in tCO2e",
    )
    total_change_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total change in tCO2e",
    )
    total_change_pct: Decimal = Field(
        default=Decimal("0"),
        description="Total change as percentage of base year",
    )

    @field_validator("current_year")
    @classmethod
    def _current_gte_base(cls, v: int, info: Any) -> int:
        """Validate current_year >= base_year."""
        base = info.data.get("base_year")
        if base is not None and v < base:
            raise ValueError(
                f"current_year ({v}) must be >= base_year ({base})"
            )
        return v

# ---------------------------------------------------------------------------
# 25. ProvenanceRecord
# ---------------------------------------------------------------------------

class ProvenanceRecord(GreenLangBase):
    """Provenance tracking record for a single pipeline stage.

    Records the SHA-256 hashes of inputs and outputs at each
    pipeline stage for complete audit trail and reproducibility.

    Attributes:
        stage: Pipeline stage name.
        input_hash: SHA-256 hash of stage input data.
        output_hash: SHA-256 hash of stage output data.
        timestamp: UTC timestamp of stage execution.
        agent_id: Agent identifier that executed the stage.
        version: Agent version string.
        duration_ms: Stage execution duration in milliseconds.
        record_count: Number of records processed in stage.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    stage: PipelineStage = Field(
        ...,
        description="Pipeline stage name",
    )
    input_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="SHA-256 hash of stage input data",
    )
    output_hash: str = Field(
        ...,
        min_length=64,
        max_length=128,
        description="SHA-256 hash of stage output data",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of stage execution",
    )
    agent_id: str = Field(
        default=AGENT_ID,
        max_length=50,
        description="Agent identifier that executed the stage",
    )
    version: str = Field(
        default=VERSION,
        max_length=20,
        description="Agent version string",
    )
    duration_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Stage execution duration in milliseconds",
    )
    record_count: int = Field(
        default=0,
        ge=0,
        description="Number of records processed in stage",
    )
