# -*- coding: utf-8 -*-
"""
Scope 2 Location-Based Emissions Agent Data Models - AGENT-MRV-009

Pydantic v2 data models for the Scope 2 Location-Based Emissions Agent SDK
covering GHG Protocol Scope 2 location-based electricity, steam, heating,
and cooling emission calculations including:
- 4 energy types (electricity, steam, heating, cooling) with per-type
  sub-classifications (steam fuel source, heating type, cooling type)
- 26 US EPA eGRID subregion emission factors (CO2, CH4, N2O in kg/MWh)
- 130+ IEA country-level electricity emission factors (tCO2/MWh)
- 27 EU member-state emission factors (tCO2/MWh)
- DEFRA UK electricity, steam, heating, cooling factors (kgCO2e/kWh)
- 50+ country-level transmission and distribution loss factors
- IPCC Tier 1/2/3 and additional calculation methodologies
- GWP values from IPCC AR4, AR5, AR6, and AR6 20-year horizon
- Unit conversions between kWh, MWh, GJ, MMBtu, and therms
- Facility metadata with grid region mapping and optional geolocation
- Batch calculation requests across multiple facilities and energy types
- Monte Carlo and analytical uncertainty quantification
- Multi-framework regulatory compliance checking
- Aggregation by facility, energy type, grid region, or time period
- SHA-256 provenance chain for complete audit trails

Enumerations (18):
    - EnergyType, EnergyUnit, GridRegionSource, CalculationMethod,
      EmissionGas, GWPSource, EmissionFactorSource, DataQualityTier,
      FacilityType, GridRegionType, TDLossMethod, TimeGranularity,
      ComplianceStatus, ReportingPeriod, ConsumptionDataSource,
      SteamType, CoolingType, HeatingType

Constants (all Decimal for zero-hallucination deterministic arithmetic):
    - GWP_VALUES: IPCC AR4/AR5/AR6/AR6_20YR GWP (CO2, CH4, N2O)
    - EGRID_FACTORS: 26 US eGRID subregions (CO2/CH4/N2O kg/MWh)
    - IEA_COUNTRY_FACTORS: 130+ countries (tCO2/MWh)
    - EU_COUNTRY_FACTORS: 27 EU member states (tCO2/MWh)
    - DEFRA_FACTORS: UK DEFRA conversion factors (kgCO2e/kWh)
    - TD_LOSS_FACTORS: 50+ country T&D loss percentages
    - STEAM_DEFAULT_EF: Steam factors by fuel source (kgCO2e/GJ)
    - HEAT_DEFAULT_EF: Heating factors by type (kgCO2e/GJ)
    - COOLING_DEFAULT_EF: Cooling factors by type (kgCO2e/GJ)
    - UNIT_CONVERSIONS: Energy unit conversion factors

Data Models (18):
    - FacilityInfo, GridRegion, GridEmissionFactor, EnergyConsumption,
      ElectricityConsumptionRequest, SteamHeatCoolingRequest,
      TransmissionLossInput, CalculationRequest, GasEmissionDetail,
      CalculationResult, BatchCalculationRequest, BatchCalculationResult,
      ComplianceCheckResult, UncertaintyRequest, UncertaintyResult,
      AggregationResult, GridFactorLookupResult, TDLossResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from greenlang.schemas import GreenLangBase, utcnow

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of calculations in a single batch request.
MAX_CALCULATIONS_PER_BATCH: int = 10_000

#: Maximum number of gas emission entries per calculation result.
MAX_GASES_PER_RESULT: int = 10

#: Maximum number of trace steps in a single calculation.
MAX_TRACE_STEPS: int = 200

#: Maximum number of facility records per tenant.
MAX_FACILITIES_PER_TENANT: int = 50_000

#: Maximum number of energy consumption records per calculation request.
MAX_ENERGY_RECORDS_PER_CALC: int = 1_000

#: Default Monte Carlo simulation iterations for uncertainty analysis.
DEFAULT_MONTE_CARLO_ITERATIONS: int = 10_000

#: Default confidence level for uncertainty intervals.
DEFAULT_CONFIDENCE_LEVEL: Decimal = Decimal("0.95")

#: Prefix for all database table names in this module.
TABLE_PREFIX: str = "gl_s2l_"

# =============================================================================
# Enumerations (18)
# =============================================================================

class EnergyType(str, Enum):
    """Classification of purchased energy types for Scope 2 reporting.

    GHG Protocol Scope 2 Guidance requires organisations to report
    indirect emissions from four categories of purchased or acquired
    energy. Each category has distinct emission factor sources and
    calculation approaches.

    ELECTRICITY: Grid-supplied electrical energy. Location-based
        emissions use grid-average emission factors from regional
        or national grids. The dominant Scope 2 source for most
        organisations.
    STEAM: Purchased steam from external suppliers or district
        steam networks. Emission factors depend on the fuel used
        to generate the steam (natural gas, coal, biomass).
    HEATING: Purchased heat from district heating networks or
        external boiler systems. Factors vary by heating source
        (district, gas boiler, electric).
    COOLING: Purchased chilled water or cooling from district
        cooling networks. Factors vary by cooling technology
        (electric chiller, absorption, district).
    """

    ELECTRICITY = "electricity"
    STEAM = "steam"
    HEATING = "heating"
    COOLING = "cooling"

class EnergyUnit(str, Enum):
    """Units of measurement for energy consumption quantities.

    All energy consumption data must be expressed in a known unit
    for conversion to a common basis (MWh for electricity, GJ for
    steam/heat/cooling) before applying emission factors.

    KWH: Kilowatt-hours. SI derived unit. 1 kWh = 3.6 MJ.
    MWH: Megawatt-hours. 1 MWh = 1000 kWh = 3.6 GJ.
        Standard basis for electricity emission factors.
    GJ: Gigajoules. SI unit. 1 GJ = 277.778 kWh.
        Standard basis for steam, heat, and cooling factors.
    MMBTU: Million British Thermal Units. 1 MMBtu = 1.05506 GJ.
        Common in US natural gas and steam metering.
    THERMS: Therms. 1 therm = 100,000 BTU = 0.105506 GJ.
        Common in US residential and commercial gas billing.
    """

    KWH = "kwh"
    MWH = "mwh"
    GJ = "gj"
    MMBTU = "mmbtu"
    THERMS = "therms"

class GridRegionSource(str, Enum):
    """Source authority for grid region definitions and boundaries.

    Identifies the authoritative database or registry that defines
    the geographic boundaries of a grid region and publishes the
    associated electricity emission factors.

    EGRID: US EPA Emissions & Generation Resource Integrated Database.
        Defines 26 subregions within the contiguous United States,
        Alaska, and Hawaii. Updated annually.
    IEA: International Energy Agency. Publishes country-level
        electricity CO2 emission factors for 130+ countries.
    EU_EEA: European Environment Agency. Publishes EU member-state
        electricity emission factors under the EEA greenhouse gas
        data viewer.
    DEFRA: UK Department for Environment, Food and Rural Affairs.
        Publishes UK-specific electricity and energy conversion
        factors updated annually.
    NATIONAL: Country-specific national inventory or grid operator
        data not covered by the above international sources.
    CUSTOM: User-defined grid region with custom emission factors
        supported by documented evidence.
    """

    EGRID = "egrid"
    IEA = "iea"
    EU_EEA = "eu_eea"
    DEFRA = "defra"
    NATIONAL = "national"
    CUSTOM = "custom"

class CalculationMethod(str, Enum):
    """Methodology for calculating Scope 2 location-based emissions.

    IPCC_TIER_1: Simplest approach using published grid-average
        emission factors (country or regional level) applied to
        total energy consumption. No facility-specific adjustments.
    IPCC_TIER_2: Uses subregional emission factors (e.g. eGRID
        subregion) and may incorporate monthly or seasonal variation
        in grid emission intensity.
    IPCC_TIER_3: Uses time-resolved (hourly or sub-hourly) marginal
        or average emission factors from grid operators or real-time
        dispatch data. Highest accuracy, requires granular consumption
        metering.
    MASS_BALANCE: Applies a mass-balance approach to steam, heating,
        and cooling by tracking the fuel input to the energy
        generation system and the system efficiency.
    DIRECT_MEASUREMENT: Uses continuous emissions monitoring system
        (CEMS) data from the energy supplier, allocated to the
        purchaser based on metered consumption.
    SPEND_BASED: Estimates emissions from energy expenditure data
        using average cost per unit of energy and emission factors.
        Lowest accuracy, used only when consumption data is
        unavailable.
    """

    IPCC_TIER_1 = "ipcc_tier_1"
    IPCC_TIER_2 = "ipcc_tier_2"
    IPCC_TIER_3 = "ipcc_tier_3"
    MASS_BALANCE = "mass_balance"
    DIRECT_MEASUREMENT = "direct_measurement"
    SPEND_BASED = "spend_based"

class EmissionGas(str, Enum):
    """Greenhouse gases tracked in Scope 2 location-based calculations.

    CO2: Carbon dioxide. Primary emission from fossil-fuel-based
        electricity generation and steam production. Accounts for
        the vast majority of Scope 2 emissions.
    CH4: Methane. Emitted from upstream fuel extraction and
        incomplete combustion at power plants. Small fraction of
        total Scope 2 but included for completeness per GHG Protocol.
    N2O: Nitrous oxide. Emitted from combustion processes at power
        plants, particularly coal and biomass. Included for full
        GHG Protocol compliance.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"

class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for Global Warming Potential values.

    Determines which set of 100-year (or 20-year) GWP multipliers to
    apply when converting individual gas emissions (CH4, N2O) to CO2
    equivalent totals.

    AR4: Fourth Assessment Report (2007). 100-year GWP.
        CO2=1, CH4=25, N2O=298.
    AR5: Fifth Assessment Report (2014). 100-year GWP.
        CO2=1, CH4=28, N2O=265.
    AR6: Sixth Assessment Report (2021). 100-year GWP.
        CO2=1, CH4=27.9, N2O=273.
    AR6_20YR: Sixth Assessment Report (2021). 20-year GWP.
        CO2=1, CH4=81.2, N2O=273. Highlights near-term climate
        impact of short-lived pollutants.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"

class EmissionFactorSource(str, Enum):
    """Authoritative source for Scope 2 grid emission factors.

    EGRID: US EPA eGRID database. Provides subregional CO2, CH4,
        and N2O emission rates for US electricity generation.
    IEA: International Energy Agency CO2 Emissions from Fuel
        Combustion. Country-level electricity emission factors.
    DEFRA: UK DEFRA Greenhouse Gas Reporting conversion factors.
        UK-specific electricity, steam, heating, and cooling factors.
    EU_EEA: European Environment Agency greenhouse gas data viewer.
        EU member-state electricity emission factors.
    NATIONAL: Country-specific national inventory factors published
        by national statistical offices or grid operators.
    CUSTOM: User-provided emission factors with documented provenance.
        Requires evidence attachment for audit compliance.
    IPCC: IPCC Guidelines for National Greenhouse Gas Inventories.
        Default emission factors when country-specific data is
        unavailable.
    """

    EGRID = "egrid"
    IEA = "iea"
    DEFRA = "defra"
    EU_EEA = "eu_eea"
    NATIONAL = "national"
    CUSTOM = "custom"
    IPCC = "ipcc"

class DataQualityTier(str, Enum):
    """Data quality classification for emission factor inputs.

    Tier classification follows the IPCC approach to data quality
    and determines the uncertainty range applied to calculation
    results.

    TIER_1: Default values from international databases (IPCC, IEA).
        Highest uncertainty range (+/- 25-50%). Used when no
        country-specific or regional data is available.
    TIER_2: Country-specific or subregional data (eGRID subregion,
        national inventory). Moderate uncertainty (+/- 10-25%).
    TIER_3: Facility-specific or utility-specific data from CEMS
        or supplier disclosure. Lowest uncertainty (+/- 5-10%).
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"

class FacilityType(str, Enum):
    """Classification of reporting facilities by primary function.

    Determines default energy intensity benchmarks and applicable
    reporting templates. Used for aggregation and comparative
    analysis across facility portfolios.

    OFFICE: Commercial office building or co-working space.
    WAREHOUSE: Warehousing, distribution center, or logistics hub.
    MANUFACTURING: Industrial manufacturing or production facility.
    RETAIL: Retail store, shopping center, or commercial outlet.
    DATA_CENTER: IT data center or colocation facility. Typically
        high electricity intensity with significant cooling load.
    HOSPITAL: Healthcare facility including hospitals, clinics,
        and medical research centers.
    SCHOOL: Educational institution including schools, universities,
        and research campuses.
    OTHER: Facility types not classified in the above categories.
    """

    OFFICE = "office"
    WAREHOUSE = "warehouse"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    DATA_CENTER = "data_center"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    OTHER = "other"

class GridRegionType(str, Enum):
    """Geographic granularity of grid region definitions.

    Determines the level of spatial resolution for emission factor
    lookup. Finer granularity generally provides more accurate
    location-based calculations.

    COUNTRY: National-level grid region (e.g. IEA country factors).
        Coarsest resolution, used for most international reporting.
    SUBREGION: Multi-state or multi-province subregion (e.g. eGRID
        subregion CAMX covering California and parts of the
        southwestern US).
    STATE: State, province, or territory-level region. Available
        for countries with state-level grid data.
    CUSTOM: User-defined region boundaries with custom emission
        factors. Must include documented boundary definition.
    """

    COUNTRY = "country"
    SUBREGION = "subregion"
    STATE = "state"
    CUSTOM = "custom"

class TDLossMethod(str, Enum):
    """Method for estimating transmission and distribution (T&D) losses.

    T&D losses represent electricity lost in the grid between the
    point of generation and the point of consumption. GHG Protocol
    Scope 2 Guidance requires reporting of T&D loss emissions
    separately or as part of location-based totals.

    COUNTRY_AVERAGE: Uses published national average T&D loss
        percentages (World Bank or IEA data). Most common approach.
    REGIONAL: Uses regional or utility-specific T&D loss data
        when available from grid operators or regulators.
    CUSTOM: User-provided T&D loss percentage with documented
        evidence from the specific utility or grid operator.
    """

    COUNTRY_AVERAGE = "country_average"
    REGIONAL = "regional"
    CUSTOM = "custom"

class TimeGranularity(str, Enum):
    """Temporal resolution for energy consumption data and calculations.

    Determines whether emission factors are applied at annual,
    monthly, or hourly resolution. Finer granularity captures
    temporal variation in grid emission intensity.

    ANNUAL: Single annual emission factor applied to total annual
        consumption. Simplest approach (IPCC Tier 1).
    MONTHLY: Monthly emission factors applied to monthly consumption
        data. Captures seasonal variation in generation mix.
    HOURLY: Hourly marginal or average emission factors applied to
        hourly consumption data. Highest accuracy, requires
        interval metering (IPCC Tier 3).
    """

    ANNUAL = "annual"
    MONTHLY = "monthly"
    HOURLY = "hourly"

class ComplianceStatus(str, Enum):
    """Result of a regulatory compliance check for a calculation.

    COMPLIANT: All requirements of the regulatory framework are
        fully satisfied for the given calculation.
    NON_COMPLIANT: One or more mandatory requirements are not met.
    PARTIAL: Some requirements are met but others are missing,
        incomplete, or require additional evidence.
    NOT_ASSESSED: Compliance has not been evaluated against this
        particular regulatory framework.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_ASSESSED = "not_assessed"

class ReportingPeriod(str, Enum):
    """Time period for emission aggregation and reporting outputs.

    ANNUAL: Full calendar or fiscal year. Standard for CDP, CSRD,
        and GHG Protocol corporate inventories.
    QUARTERLY: Calendar quarter (Q1-Q4). Used for interim
        management reporting and progress tracking.
    MONTHLY: Calendar month. Used for operational dashboards
        and trend analysis.
    CUSTOM: User-defined date range with explicit start and end
        dates specified in the aggregation request.
    """

    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class ConsumptionDataSource(str, Enum):
    """Origin of energy consumption data for data quality assessment.

    The data source classification determines the uncertainty
    range assigned to the consumption quantity and influences
    the overall data quality tier of the calculation.

    METER: Direct meter reading from a revenue-grade utility meter
        or sub-meter. Highest accuracy and lowest uncertainty.
    INVOICE: Consumption quantity from utility invoices or bills.
        High accuracy but may include estimated reads during
        billing cycles.
    ESTIMATE: Engineering estimate or extrapolation from partial
        data. Moderate uncertainty. Must be documented with
        estimation methodology.
    BENCHMARK: Derived from floor area, headcount, or industry
        benchmarks (e.g. CBECS, CIBSE TM46). Highest uncertainty.
        Used only when no measured or invoiced data is available.
    """

    METER = "meter"
    INVOICE = "invoice"
    ESTIMATE = "estimate"
    BENCHMARK = "benchmark"

class SteamType(str, Enum):
    """Classification of steam generation fuel source.

    Determines the default emission factor applied to purchased
    steam consumption when supplier-specific data is unavailable.

    NATURAL_GAS: Steam generated from natural gas combustion.
        Lowest emission intensity among fossil-fuel steam sources.
    COAL: Steam generated from coal combustion. Highest emission
        intensity among common steam generation fuels.
    BIOMASS: Steam generated from biomass combustion. Biogenic
        CO2 is reported separately; fossil CO2e is zero for
        sustainable biomass per GHG Protocol guidance.
    """

    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    BIOMASS = "biomass"

class CoolingType(str, Enum):
    """Classification of cooling system technology.

    Determines the default emission factor applied to purchased
    cooling consumption when supplier-specific data is unavailable.

    ELECTRIC_CHILLER: Vapour-compression electric chiller. Emission
        factor varies by grid electricity emission intensity and
        chiller coefficient of performance (COP).
    ABSORPTION: Absorption chiller driven by heat (natural gas,
        steam, or waste heat). Lower electricity consumption but
        higher direct fuel emissions.
    DISTRICT: District cooling network. Emission factor depends
        on the central plant technology mix and network losses.
    """

    ELECTRIC_CHILLER = "electric_chiller"
    ABSORPTION = "absorption"
    DISTRICT = "district"

class HeatingType(str, Enum):
    """Classification of heating system technology.

    Determines the default emission factor applied to purchased
    heating consumption when supplier-specific data is unavailable.

    DISTRICT: District heating network supplied by a central plant
        (CHP, boiler plant, waste heat recovery). Emission factor
        depends on the district heating fuel mix and network losses.
    GAS_BOILER: On-site or external natural gas boiler. Emission
        factor based on natural gas combustion intensity and
        boiler efficiency.
    ELECTRIC: Electric resistance or heat pump heating. Emission
        factor varies by grid electricity emission intensity and
        system efficiency (COP for heat pumps).
    """

    DISTRICT = "district"
    GAS_BOILER = "gas_boiler"
    ELECTRIC = "electric"

# =============================================================================
# Constant Tables (all Decimal for deterministic arithmetic)
# =============================================================================

# ---------------------------------------------------------------------------
# GWP values by IPCC Assessment Report
# ---------------------------------------------------------------------------

#: Global Warming Potential values for Scope 2 greenhouse gases by IPCC AR.
#: Units: dimensionless multiplier (kg CO2e per kg gas).
#: Decimal precision for zero-hallucination regulatory calculations.
#: Sources:
#:   AR4: IPCC Fourth Assessment Report (2007), Table 2.14.
#:   AR5: IPCC Fifth Assessment Report (2014), Table 8.A.1.
#:   AR6: IPCC Sixth Assessment Report (2021), Table 7.15.
#:   AR6_20YR: AR6 GWP-20yr timeframe, Table 7.15.
GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": Decimal("1"),
        "CH4": Decimal("25"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CO2": Decimal("1"),
        "CH4": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CO2": Decimal("1"),
        "CH4": Decimal("27.9"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("81.2"),
        "N2O": Decimal("273"),
    },
}

# ---------------------------------------------------------------------------
# US EPA eGRID Subregion Emission Factors
# ---------------------------------------------------------------------------

#: EPA eGRID subregion average emission rates for electricity generation.
#: Units: kg per MWh of generation (CO2 in kg/MWh, CH4 in kg/MWh,
#:         N2O in kg/MWh).
#: Source: US EPA eGRID2022 (released January 2024), Table 1.
#:         Annual output emission rates for the 26 eGRID subregions.
#:
#: Each entry is keyed by the eGRID subregion acronym and contains
#: Decimal values for CO2, CH4, and N2O emission rates. These rates
#: represent the average emission intensity of electricity generated
#: within each subregion, reflecting the fuel mix of generators.
#:
#: To calculate location-based Scope 2 emissions:
#:   emissions_kg = consumption_mwh * ef_kg_per_mwh
#:
#: Note: Emission factors are at the generation level. T&D losses
#: should be applied separately using TD_LOSS_FACTORS to account
#: for grid losses between generation and consumption.
EGRID_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "AKGD": {
        "CO2": Decimal("442.80"),
        "CH4": Decimal("0.044"),
        "N2O": Decimal("0.006"),
    },
    "AKMS": {
        "CO2": Decimal("195.50"),
        "CH4": Decimal("0.020"),
        "N2O": Decimal("0.003"),
    },
    "AZNM": {
        "CO2": Decimal("370.40"),
        "CH4": Decimal("0.037"),
        "N2O": Decimal("0.005"),
    },
    "CAMX": {
        "CO2": Decimal("225.30"),
        "CH4": Decimal("0.026"),
        "N2O": Decimal("0.003"),
    },
    "ERCT": {
        "CO2": Decimal("380.10"),
        "CH4": Decimal("0.038"),
        "N2O": Decimal("0.005"),
    },
    "FRCC": {
        "CO2": Decimal("389.20"),
        "CH4": Decimal("0.039"),
        "N2O": Decimal("0.005"),
    },
    "HIMS": {
        "CO2": Decimal("505.60"),
        "CH4": Decimal("0.050"),
        "N2O": Decimal("0.007"),
    },
    "HIOA": {
        "CO2": Decimal("641.30"),
        "CH4": Decimal("0.064"),
        "N2O": Decimal("0.009"),
    },
    "MROE": {
        "CO2": Decimal("580.60"),
        "CH4": Decimal("0.058"),
        "N2O": Decimal("0.008"),
    },
    "MROW": {
        "CO2": Decimal("452.70"),
        "CH4": Decimal("0.045"),
        "N2O": Decimal("0.006"),
    },
    "NEWE": {
        "CO2": Decimal("213.40"),
        "CH4": Decimal("0.027"),
        "N2O": Decimal("0.003"),
    },
    "NWPP": {
        "CO2": Decimal("280.50"),
        "CH4": Decimal("0.028"),
        "N2O": Decimal("0.004"),
    },
    "NYCW": {
        "CO2": Decimal("244.60"),
        "CH4": Decimal("0.024"),
        "N2O": Decimal("0.003"),
    },
    "NYLI": {
        "CO2": Decimal("476.30"),
        "CH4": Decimal("0.048"),
        "N2O": Decimal("0.006"),
    },
    "NYUP": {
        "CO2": Decimal("115.30"),
        "CH4": Decimal("0.015"),
        "N2O": Decimal("0.002"),
    },
    "PRMS": {
        "CO2": Decimal("613.80"),
        "CH4": Decimal("0.061"),
        "N2O": Decimal("0.008"),
    },
    "RFCE": {
        "CO2": Decimal("300.20"),
        "CH4": Decimal("0.030"),
        "N2O": Decimal("0.004"),
    },
    "RFCM": {
        "CO2": Decimal("580.90"),
        "CH4": Decimal("0.058"),
        "N2O": Decimal("0.008"),
    },
    "RFCW": {
        "CO2": Decimal("470.50"),
        "CH4": Decimal("0.047"),
        "N2O": Decimal("0.006"),
    },
    "RMPA": {
        "CO2": Decimal("548.30"),
        "CH4": Decimal("0.055"),
        "N2O": Decimal("0.007"),
    },
    "SPNO": {
        "CO2": Decimal("529.10"),
        "CH4": Decimal("0.053"),
        "N2O": Decimal("0.007"),
    },
    "SPSO": {
        "CO2": Decimal("420.80"),
        "CH4": Decimal("0.042"),
        "N2O": Decimal("0.006"),
    },
    "SRMV": {
        "CO2": Decimal("349.70"),
        "CH4": Decimal("0.035"),
        "N2O": Decimal("0.005"),
    },
    "SRMW": {
        "CO2": Decimal("629.40"),
        "CH4": Decimal("0.063"),
        "N2O": Decimal("0.008"),
    },
    "SRSO": {
        "CO2": Decimal("395.00"),
        "CH4": Decimal("0.040"),
        "N2O": Decimal("0.005"),
    },
    "SRTV": {
        "CO2": Decimal("388.60"),
        "CH4": Decimal("0.039"),
        "N2O": Decimal("0.005"),
    },
    "SRVC": {
        "CO2": Decimal("298.10"),
        "CH4": Decimal("0.030"),
        "N2O": Decimal("0.004"),
    },
}

# ---------------------------------------------------------------------------
# IEA Country-Level Electricity Emission Factors
# ---------------------------------------------------------------------------

#: IEA country-level CO2 emission factors for electricity generation.
#: Units: tonnes CO2 per MWh (tCO2/MWh).
#: Source: IEA CO2 Emissions from Fuel Combustion (2023 edition).
#:
#: Factors represent the CO2 intensity of national electricity
#: generation mixes. For CH4 and N2O, apply IPCC default ratios
#: or use country-specific factors when available.
#:
#: Keys use ISO 3166-1 alpha-2 country codes (uppercase).
IEA_COUNTRY_FACTORS: Dict[str, Decimal] = {
    # Americas
    "US": Decimal("0.379"),
    "CA": Decimal("0.120"),
    "MX": Decimal("0.424"),
    "BR": Decimal("0.074"),
    "AR": Decimal("0.315"),
    "CL": Decimal("0.357"),
    "CO": Decimal("0.146"),
    "PE": Decimal("0.196"),
    "VE": Decimal("0.178"),
    "EC": Decimal("0.167"),
    "UY": Decimal("0.039"),
    "PY": Decimal("0.000"),
    "BO": Decimal("0.372"),
    "CR": Decimal("0.029"),
    "PA": Decimal("0.195"),
    "GT": Decimal("0.318"),
    "DO": Decimal("0.551"),
    "HN": Decimal("0.318"),
    "SV": Decimal("0.214"),
    "NI": Decimal("0.327"),
    "JM": Decimal("0.670"),
    "TT": Decimal("0.558"),
    "HT": Decimal("0.634"),
    "CU": Decimal("0.870"),
    # Europe
    "GB": Decimal("0.212"),
    "DE": Decimal("0.338"),
    "FR": Decimal("0.056"),
    "IT": Decimal("0.233"),
    "ES": Decimal("0.138"),
    "PL": Decimal("0.635"),
    "NL": Decimal("0.328"),
    "BE": Decimal("0.155"),
    "SE": Decimal("0.008"),
    "NO": Decimal("0.008"),
    "DK": Decimal("0.115"),
    "FI": Decimal("0.072"),
    "AT": Decimal("0.086"),
    "CH": Decimal("0.015"),
    "IE": Decimal("0.296"),
    "PT": Decimal("0.178"),
    "GR": Decimal("0.352"),
    "CZ": Decimal("0.395"),
    "RO": Decimal("0.265"),
    "HU": Decimal("0.217"),
    "BG": Decimal("0.374"),
    "SK": Decimal("0.101"),
    "HR": Decimal("0.157"),
    "SI": Decimal("0.214"),
    "LT": Decimal("0.036"),
    "LV": Decimal("0.099"),
    "EE": Decimal("0.579"),
    "CY": Decimal("0.592"),
    "LU": Decimal("0.079"),
    "MT": Decimal("0.391"),
    "IS": Decimal("0.000"),
    "RS": Decimal("0.619"),
    "BA": Decimal("0.712"),
    "MK": Decimal("0.546"),
    "AL": Decimal("0.013"),
    "ME": Decimal("0.335"),
    "XK": Decimal("0.916"),
    "UA": Decimal("0.296"),
    "MD": Decimal("0.489"),
    "BY": Decimal("0.355"),
    "GE": Decimal("0.109"),
    # Asia-Pacific
    "CN": Decimal("0.555"),
    "JP": Decimal("0.457"),
    "IN": Decimal("0.708"),
    "KR": Decimal("0.415"),
    "AU": Decimal("0.656"),
    "NZ": Decimal("0.083"),
    "TW": Decimal("0.502"),
    "TH": Decimal("0.432"),
    "MY": Decimal("0.550"),
    "SG": Decimal("0.408"),
    "ID": Decimal("0.712"),
    "PH": Decimal("0.543"),
    "VN": Decimal("0.477"),
    "BD": Decimal("0.576"),
    "PK": Decimal("0.354"),
    "LK": Decimal("0.392"),
    "MM": Decimal("0.377"),
    "KH": Decimal("0.580"),
    "LA": Decimal("0.223"),
    "NP": Decimal("0.000"),
    "MN": Decimal("0.707"),
    "KZ": Decimal("0.563"),
    "UZ": Decimal("0.458"),
    "TM": Decimal("0.743"),
    "KG": Decimal("0.062"),
    "TJ": Decimal("0.031"),
    # Middle East
    "SA": Decimal("0.583"),
    "AE": Decimal("0.397"),
    "QA": Decimal("0.396"),
    "KW": Decimal("0.560"),
    "BH": Decimal("0.546"),
    "OM": Decimal("0.433"),
    "IL": Decimal("0.490"),
    "JO": Decimal("0.427"),
    "LB": Decimal("0.670"),
    "IQ": Decimal("0.705"),
    "IR": Decimal("0.489"),
    "YE": Decimal("0.694"),
    "SY": Decimal("0.608"),
    # Africa
    "ZA": Decimal("0.928"),
    "EG": Decimal("0.420"),
    "NG": Decimal("0.407"),
    "MA": Decimal("0.610"),
    "DZ": Decimal("0.441"),
    "TN": Decimal("0.408"),
    "LY": Decimal("0.648"),
    "KE": Decimal("0.093"),
    "GH": Decimal("0.253"),
    "ET": Decimal("0.000"),
    "TZ": Decimal("0.333"),
    "CI": Decimal("0.335"),
    "SN": Decimal("0.522"),
    "CM": Decimal("0.162"),
    "ZW": Decimal("0.643"),
    "ZM": Decimal("0.021"),
    "MZ": Decimal("0.039"),
    "AO": Decimal("0.227"),
    "SD": Decimal("0.553"),
    "CD": Decimal("0.003"),
    "UG": Decimal("0.025"),
    "RW": Decimal("0.328"),
    "MU": Decimal("0.600"),
    "NA": Decimal("0.135"),
    "BW": Decimal("1.019"),
    "MW": Decimal("0.059"),
    "MG": Decimal("0.440"),
    "BJ": Decimal("0.594"),
    "GA": Decimal("0.370"),
    # Central Asia / Other
    "TR": Decimal("0.388"),
    "RU": Decimal("0.340"),
    "AZ": Decimal("0.399"),
    "AM": Decimal("0.147"),
}

# ---------------------------------------------------------------------------
# EU Member-State Electricity Emission Factors
# ---------------------------------------------------------------------------

#: EU-27 member state electricity CO2 emission factors.
#: Units: tonnes CO2 per MWh (tCO2/MWh).
#: Source: European Environment Agency (EEA) greenhouse gas data viewer,
#:         2023 reporting year.
#:
#: Keys use ISO 3166-1 alpha-2 country codes (uppercase).
#: These factors are specific to EU regulatory reporting under CSRD
#: and EU ETS. For non-EU reporting, use IEA_COUNTRY_FACTORS.
EU_COUNTRY_FACTORS: Dict[str, Decimal] = {
    "AT": Decimal("0.086"),
    "BE": Decimal("0.155"),
    "BG": Decimal("0.374"),
    "HR": Decimal("0.157"),
    "CY": Decimal("0.592"),
    "CZ": Decimal("0.395"),
    "DK": Decimal("0.115"),
    "EE": Decimal("0.579"),
    "FI": Decimal("0.072"),
    "FR": Decimal("0.056"),
    "DE": Decimal("0.338"),
    "GR": Decimal("0.352"),
    "HU": Decimal("0.217"),
    "IE": Decimal("0.296"),
    "IT": Decimal("0.233"),
    "LV": Decimal("0.099"),
    "LT": Decimal("0.036"),
    "LU": Decimal("0.079"),
    "MT": Decimal("0.391"),
    "NL": Decimal("0.328"),
    "PL": Decimal("0.635"),
    "PT": Decimal("0.178"),
    "RO": Decimal("0.265"),
    "SK": Decimal("0.101"),
    "SI": Decimal("0.214"),
    "ES": Decimal("0.138"),
    "SE": Decimal("0.008"),
}

# ---------------------------------------------------------------------------
# UK DEFRA Conversion Factors
# ---------------------------------------------------------------------------

#: UK DEFRA greenhouse gas conversion factors for company reporting.
#: Units: kgCO2e per kWh.
#: Source: DEFRA/BEIS UK Government GHG Conversion Factors for
#:         Company Reporting (2024 edition).
#:
#: electricity_generation: Grid electricity generation factor only.
#: electricity_td: Transmission and distribution loss factor only.
#: electricity_total: Sum of generation + T&D factors.
#: steam: Purchased steam conversion factor.
#: heating: Purchased heat conversion factor.
#: cooling: Purchased cooling conversion factor.
DEFRA_FACTORS: Dict[str, Decimal] = {
    "electricity_generation": Decimal("0.20707"),
    "electricity_td": Decimal("0.01879"),
    "electricity_total": Decimal("0.22586"),
    "steam": Decimal("0.07050"),
    "heating": Decimal("0.04350"),
    "cooling": Decimal("0.03210"),
}

# ---------------------------------------------------------------------------
# Transmission & Distribution Loss Factors
# ---------------------------------------------------------------------------

#: Country-level electricity transmission and distribution loss
#: percentages (fraction of total generation lost in the grid).
#: Units: dimensionless fraction (0.0 - 1.0).
#: Source: World Bank World Development Indicators (EG.ELC.LOSS.ZS)
#:         and IEA Electricity Information.
#:
#: T&D losses = gross_generation * td_loss_factor
#: Net consumption = gross_generation * (1 - td_loss_factor)
#:
#: Keys use ISO 3166-1 alpha-2 country codes (uppercase).
TD_LOSS_FACTORS: Dict[str, Decimal] = {
    # Americas
    "US": Decimal("0.050"),
    "CA": Decimal("0.070"),
    "MX": Decimal("0.138"),
    "BR": Decimal("0.156"),
    "AR": Decimal("0.128"),
    "CL": Decimal("0.065"),
    "CO": Decimal("0.120"),
    "PE": Decimal("0.110"),
    "VE": Decimal("0.270"),
    "EC": Decimal("0.130"),
    "UY": Decimal("0.098"),
    "PY": Decimal("0.232"),
    "BO": Decimal("0.095"),
    "CR": Decimal("0.088"),
    "PA": Decimal("0.117"),
    # Europe
    "GB": Decimal("0.077"),
    "DE": Decimal("0.040"),
    "FR": Decimal("0.060"),
    "IT": Decimal("0.063"),
    "ES": Decimal("0.089"),
    "PL": Decimal("0.063"),
    "NL": Decimal("0.040"),
    "BE": Decimal("0.048"),
    "SE": Decimal("0.065"),
    "NO": Decimal("0.062"),
    "DK": Decimal("0.055"),
    "FI": Decimal("0.033"),
    "AT": Decimal("0.055"),
    "CH": Decimal("0.058"),
    "IE": Decimal("0.078"),
    "PT": Decimal("0.089"),
    "GR": Decimal("0.068"),
    "CZ": Decimal("0.055"),
    "RO": Decimal("0.115"),
    "HU": Decimal("0.100"),
    "BG": Decimal("0.092"),
    "SK": Decimal("0.031"),
    "HR": Decimal("0.089"),
    "SI": Decimal("0.053"),
    "LT": Decimal("0.088"),
    "LV": Decimal("0.074"),
    "EE": Decimal("0.064"),
    "CY": Decimal("0.045"),
    "LU": Decimal("0.062"),
    "MT": Decimal("0.100"),
    # Asia-Pacific
    "CN": Decimal("0.058"),
    "JP": Decimal("0.050"),
    "IN": Decimal("0.194"),
    "KR": Decimal("0.036"),
    "AU": Decimal("0.055"),
    "NZ": Decimal("0.065"),
    "TW": Decimal("0.042"),
    "TH": Decimal("0.061"),
    "MY": Decimal("0.045"),
    "SG": Decimal("0.023"),
    "ID": Decimal("0.096"),
    "PH": Decimal("0.096"),
    "VN": Decimal("0.092"),
    "BD": Decimal("0.112"),
    "PK": Decimal("0.178"),
    # Middle East
    "SA": Decimal("0.073"),
    "AE": Decimal("0.065"),
    "QA": Decimal("0.048"),
    "KW": Decimal("0.056"),
    "IL": Decimal("0.037"),
    "TR": Decimal("0.118"),
    # Africa
    "ZA": Decimal("0.084"),
    "EG": Decimal("0.120"),
    "NG": Decimal("0.216"),
    "MA": Decimal("0.118"),
    "KE": Decimal("0.184"),
    "GH": Decimal("0.183"),
    "TZ": Decimal("0.165"),
    # Russia and Central Asia
    "RU": Decimal("0.105"),
    "KZ": Decimal("0.110"),
    "UZ": Decimal("0.098"),
}

# ---------------------------------------------------------------------------
# Steam Default Emission Factors
# ---------------------------------------------------------------------------

#: Default emission factors for purchased steam by fuel source.
#: Units: kgCO2e per GJ of steam energy content.
#: Source: GHG Protocol Scope 2 Guidance, IPCC 2006 Guidelines Vol 2.
#:
#: natural_gas: Steam produced from natural gas-fired boilers.
#: coal: Steam produced from coal-fired boilers.
#: biomass: Steam from sustainable biomass (biogenic CO2 = 0).
#: oil: Steam produced from fuel-oil-fired boilers.
STEAM_DEFAULT_EF: Dict[str, Decimal] = {
    "natural_gas": Decimal("56.10"),
    "coal": Decimal("94.60"),
    "biomass": Decimal("0.00"),
    "oil": Decimal("73.30"),
}

# ---------------------------------------------------------------------------
# Heat Default Emission Factors
# ---------------------------------------------------------------------------

#: Default emission factors for purchased heat by heating type.
#: Units: kgCO2e per GJ of heat energy content.
#: Source: GHG Protocol Scope 2 Guidance, DEFRA/BEIS factors.
#:
#: district: District heating network average.
#: gas_boiler: Natural gas boiler heat.
#: electric: Electric heating (varies by grid; placeholder value
#:     used only as a fallback when grid factor is unavailable).
HEAT_DEFAULT_EF: Dict[str, Decimal] = {
    "district": Decimal("43.50"),
    "gas_boiler": Decimal("56.10"),
    "electric": Decimal("0.00"),
}

# ---------------------------------------------------------------------------
# Cooling Default Emission Factors
# ---------------------------------------------------------------------------

#: Default emission factors for purchased cooling by cooling type.
#: Units: kgCO2e per GJ of cooling energy content.
#: Source: GHG Protocol Scope 2 Guidance, DEFRA/BEIS factors.
#:
#: electric_chiller: Electric chiller cooling (varies by grid;
#:     placeholder value used only as a fallback).
#: absorption: Absorption chiller cooling (natural gas or waste heat).
#: district: District cooling network average.
COOLING_DEFAULT_EF: Dict[str, Decimal] = {
    "electric_chiller": Decimal("0.00"),
    "absorption": Decimal("32.10"),
    "district": Decimal("28.50"),
}

# ---------------------------------------------------------------------------
# Energy Unit Conversion Factors
# ---------------------------------------------------------------------------

#: Conversion factors between energy units.
#: All values are exact Decimal representations for zero-hallucination
#: deterministic arithmetic. No floating-point rounding errors.
#:
#: MWH_TO_GJ: 1 MWh = 3.6 GJ (exact).
#: GJ_TO_MWH: 1 GJ = 0.277778 MWh (1/3.6, rounded to 6 dp).
#: MMBTU_TO_GJ: 1 MMBtu = 1.05506 GJ (EPA conversion).
#: GJ_TO_MMBTU: 1 GJ = 0.947817 MMBtu (reciprocal).
#: THERM_TO_GJ: 1 therm = 0.105506 GJ (MMBtu/10).
#: GJ_TO_THERM: 1 GJ = 9.47817 therms (reciprocal).
#: KWH_TO_MWH: 1 kWh = 0.001 MWh (exact).
#: MWH_TO_KWH: 1 MWh = 1000 kWh (exact).
UNIT_CONVERSIONS: Dict[str, Decimal] = {
    "MWH_TO_GJ": Decimal("3.6"),
    "GJ_TO_MWH": Decimal("0.277778"),
    "MMBTU_TO_GJ": Decimal("1.05506"),
    "GJ_TO_MMBTU": Decimal("0.947817"),
    "THERM_TO_GJ": Decimal("0.105506"),
    "GJ_TO_THERM": Decimal("9.47817"),
    "KWH_TO_MWH": Decimal("0.001"),
    "MWH_TO_KWH": Decimal("1000"),
}

# =============================================================================
# Data Models (18)
# =============================================================================

class FacilityInfo(GreenLangBase):
    """Metadata record for a reporting facility in the Scope 2 inventory.

    Represents a single physical facility (building, campus, or site)
    for which Scope 2 emissions are calculated. Each facility is mapped
    to a grid region for emission factor lookup and supports optional
    geolocation for spatial analysis.

    Attributes:
        facility_id: Unique system identifier for the facility (UUID).
        name: Human-readable facility name or label.
        facility_type: Classification of facility by primary function.
        country_code: ISO 3166-1 alpha-2 country code for the facility.
        grid_region_id: Identifier of the grid region (e.g. eGRID
            subregion code or IEA country code) used for emission
            factor lookup.
        egrid_subregion: US eGRID subregion code (e.g. CAMX, ERCT).
            Applicable only for US facilities.
        latitude: WGS84 latitude in decimal degrees. Optional for
            spatial analysis and grid region validation.
        longitude: WGS84 longitude in decimal degrees. Optional for
            spatial analysis and grid region validation.
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    facility_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique system identifier for the facility (UUID)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable facility name",
    )
    facility_type: FacilityType = Field(
        ...,
        description="Classification of facility by primary function",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    grid_region_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Grid region identifier for EF lookup",
    )
    egrid_subregion: Optional[str] = Field(
        default=None,
        max_length=10,
        description="US eGRID subregion code (e.g. CAMX, ERCT)",
    )
    latitude: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="WGS84 latitude in decimal degrees",
    )
    longitude: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="WGS84 longitude in decimal degrees",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Owning tenant identifier for multi-tenancy",
    )

    @field_validator("country_code")
    @classmethod
    def _uppercase_country_code(cls, v: str) -> str:
        """Normalise country code to uppercase."""
        return v.upper()

    @field_validator("egrid_subregion")
    @classmethod
    def _uppercase_egrid(cls, v: Optional[str]) -> Optional[str]:
        """Normalise eGRID subregion code to uppercase."""
        if v is not None:
            return v.upper()
        return v

class GridRegion(GreenLangBase):
    """Definition of a geographic grid region for emission factor mapping.

    Represents a named geographic area (country, subregion, state, or
    custom boundary) to which a set of grid emission factors applies.

    Attributes:
        region_id: Unique identifier for the grid region.
        name: Human-readable region name.
        region_type: Geographic granularity of the region.
        source: Authoritative source for the region definition.
        country_code: ISO 3166-1 alpha-2 country code.
        subregion_code: Sub-national region code (e.g. eGRID acronym).
        description: Optional detailed description of the region
            boundaries and applicable grid area.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    region_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the grid region",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable region name",
    )
    region_type: GridRegionType = Field(
        ...,
        description="Geographic granularity of the region",
    )
    source: GridRegionSource = Field(
        ...,
        description="Authoritative source for the region definition",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    subregion_code: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Sub-national region code (e.g. eGRID acronym)",
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed description of region boundaries",
    )

class GridEmissionFactor(GreenLangBase):
    """Grid emission factor record for a specific region, source, and year.

    Stores CO2, CH4, and N2O emission rates with calculated total
    CO2e intensity. Each factor has a data quality tier and
    provenance reference.

    Attributes:
        factor_id: Unique identifier for this emission factor record.
        region_id: Reference to the grid region.
        source: Authoritative source of the emission factor.
        year: Reporting year the factor applies to.
        co2_kg_per_mwh: CO2 emission rate in kg per MWh.
        ch4_kg_per_mwh: CH4 emission rate in kg per MWh.
        n2o_kg_per_mwh: N2O emission rate in kg per MWh.
        total_co2e_kg_per_mwh: Total CO2e emission rate in kg per
            MWh (pre-calculated using the applicable GWP values).
        data_quality_tier: Data quality classification of the factor.
        notes: Optional notes about the factor source or applicability.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    factor_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique emission factor record identifier",
    )
    region_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the grid region",
    )
    source: EmissionFactorSource = Field(
        ...,
        description="Authoritative source of the emission factor",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year the factor applies to",
    )
    co2_kg_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2 emission rate in kg per MWh",
    )
    ch4_kg_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CH4 emission rate in kg per MWh",
    )
    n2o_kg_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="N2O emission rate in kg per MWh",
    )
    total_co2e_kg_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2e emission rate in kg per MWh",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality classification of the factor",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes about the factor source",
    )

class EnergyConsumption(GreenLangBase):
    """Raw energy consumption record for a facility and reporting period.

    Represents a single consumption measurement from a meter, invoice,
    or estimate. Multiple records may exist for the same facility
    covering different energy types, meters, or time periods.

    Attributes:
        consumption_id: Unique identifier for this consumption record.
        facility_id: Reference to the consuming facility.
        energy_type: Type of purchased energy.
        quantity: Consumption quantity in the specified unit.
        unit: Unit of measurement for the quantity.
        period_start: Start date/time of the consumption period.
        period_end: End date/time of the consumption period.
        data_source: Origin of the consumption data.
        meter_id: Optional utility meter identifier for traceability.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    consumption_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique consumption record identifier",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the consuming facility",
    )
    energy_type: EnergyType = Field(
        ...,
        description="Type of purchased energy",
    )
    quantity: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Consumption quantity in the specified unit",
    )
    unit: EnergyUnit = Field(
        ...,
        description="Unit of measurement for the quantity",
    )
    period_start: datetime = Field(
        ...,
        description="Start date/time of the consumption period",
    )
    period_end: datetime = Field(
        ...,
        description="End date/time of the consumption period",
    )
    data_source: ConsumptionDataSource = Field(
        default=ConsumptionDataSource.INVOICE,
        description="Origin of the consumption data",
    )
    meter_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional utility meter identifier",
    )

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: datetime, info: Any) -> datetime:
        """Validate that period_end is after period_start."""
        start = info.data.get("period_start")
        if start is not None and v <= start:
            raise ValueError("period_end must be after period_start")
        return v

class ElectricityConsumptionRequest(GreenLangBase):
    """Request parameters for a location-based electricity emission calculation.

    Specifies the facility, consumption quantity, grid region mapping,
    and calculation options for computing Scope 2 location-based
    electricity emissions.

    Attributes:
        facility_id: Reference to the consuming facility.
        consumption_mwh: Electricity consumption in megawatt-hours.
        period_start: Start date/time of the consumption period.
        period_end: End date/time of the consumption period.
        grid_region_id: Grid region identifier for EF lookup.
            If not provided, the facility's default grid region is used.
        egrid_subregion: US eGRID subregion code. Overrides grid_region_id
            for US facilities when specified.
        country_code: ISO 3166-1 alpha-2 country code. Used as fallback
            when grid_region_id is not specified.
        gwp_source: IPCC Assessment Report for GWP values.
        include_td_losses: Whether to include T&D loss emissions.
        time_granularity: Temporal resolution for the calculation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the consuming facility",
    )
    consumption_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Electricity consumption in MWh",
    )
    period_start: datetime = Field(
        ...,
        description="Start date/time of the consumption period",
    )
    period_end: datetime = Field(
        ...,
        description="End date/time of the consumption period",
    )
    grid_region_id: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Grid region identifier for EF lookup",
    )
    egrid_subregion: Optional[str] = Field(
        default=None,
        max_length=10,
        description="US eGRID subregion code",
    )
    country_code: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    include_td_losses: bool = Field(
        default=True,
        description="Whether to include T&D loss emissions",
    )
    time_granularity: TimeGranularity = Field(
        default=TimeGranularity.ANNUAL,
        description="Temporal resolution for the calculation",
    )

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: datetime, info: Any) -> datetime:
        """Validate that period_end is after period_start."""
        start = info.data.get("period_start")
        if start is not None and v <= start:
            raise ValueError("period_end must be after period_start")
        return v

    @field_validator("egrid_subregion")
    @classmethod
    def _uppercase_egrid(cls, v: Optional[str]) -> Optional[str]:
        """Normalise eGRID subregion code to uppercase."""
        if v is not None:
            return v.upper()
        return v

    @field_validator("country_code")
    @classmethod
    def _uppercase_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Normalise country code to uppercase."""
        if v is not None:
            return v.upper()
        return v

class SteamHeatCoolingRequest(GreenLangBase):
    """Request parameters for a steam, heating, or cooling emission calculation.

    Specifies the facility, consumption quantity, energy sub-type,
    and calculation options for computing Scope 2 location-based
    emissions from purchased steam, heat, or cooling.

    Attributes:
        facility_id: Reference to the consuming facility.
        energy_type: Must be one of steam, heating, or cooling.
        consumption_gj: Energy consumption in gigajoules.
        period_start: Start date/time of the consumption period.
        period_end: End date/time of the consumption period.
        steam_type: Fuel source for steam generation. Required when
            energy_type is steam.
        heating_type: Heating technology type. Required when
            energy_type is heating.
        cooling_type: Cooling technology type. Required when
            energy_type is cooling.
        country_code: ISO 3166-1 alpha-2 country code for the facility.
        custom_ef: Optional custom emission factor in kgCO2e per GJ.
            Overrides default factors when provided.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the consuming facility",
    )
    energy_type: EnergyType = Field(
        ...,
        description="Energy type (steam, heating, or cooling)",
    )
    consumption_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Energy consumption in GJ",
    )
    period_start: datetime = Field(
        ...,
        description="Start date/time of the consumption period",
    )
    period_end: datetime = Field(
        ...,
        description="End date/time of the consumption period",
    )
    steam_type: Optional[SteamType] = Field(
        default=None,
        description="Fuel source for steam generation",
    )
    heating_type: Optional[HeatingType] = Field(
        default=None,
        description="Heating technology type",
    )
    cooling_type: Optional[CoolingType] = Field(
        default=None,
        description="Cooling technology type",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    custom_ef: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Custom emission factor in kgCO2e per GJ",
    )

    @field_validator("energy_type")
    @classmethod
    def _validate_not_electricity(cls, v: EnergyType) -> EnergyType:
        """Validate that energy_type is not electricity."""
        if v == EnergyType.ELECTRICITY:
            raise ValueError(
                "Use ElectricityConsumptionRequest for electricity; "
                "SteamHeatCoolingRequest is for steam, heating, or cooling"
            )
        return v

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: datetime, info: Any) -> datetime:
        """Validate that period_end is after period_start."""
        start = info.data.get("period_start")
        if start is not None and v <= start:
            raise ValueError("period_end must be after period_start")
        return v

    @field_validator("country_code")
    @classmethod
    def _uppercase_country_code(cls, v: str) -> str:
        """Normalise country code to uppercase."""
        return v.upper()

class TransmissionLossInput(GreenLangBase):
    """Input parameters for transmission and distribution loss calculation.

    Specifies the country, loss method, and optional custom loss
    factor for computing T&D loss emissions.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        custom_td_loss: Optional custom T&D loss percentage as a
            decimal fraction (e.g. 0.05 for 5%).
        method: Method for estimating T&D losses.
        include_upstream: Whether to include upstream T&D losses
            in the emission calculation (generation-side losses).
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    custom_td_loss: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Custom T&D loss fraction (0.0 - 1.0)",
    )
    method: TDLossMethod = Field(
        default=TDLossMethod.COUNTRY_AVERAGE,
        description="Method for estimating T&D losses",
    )
    include_upstream: bool = Field(
        default=False,
        description="Whether to include upstream T&D losses",
    )

    @field_validator("country_code")
    @classmethod
    def _uppercase_country_code(cls, v: str) -> str:
        """Normalise country code to uppercase."""
        return v.upper()

class CalculationRequest(GreenLangBase):
    """Complete request for a Scope 2 location-based emission calculation.

    Aggregates one or more electricity consumption requests and
    steam/heat/cooling requests for a single facility, along with
    shared calculation parameters.

    Attributes:
        calculation_id: Unique identifier for this calculation request.
        tenant_id: Owning tenant identifier for multi-tenancy.
        facility_id: Reference to the target facility.
        energy_requests: List of electricity consumption sub-requests.
        steam_heat_cool_requests: List of steam, heat, or cooling
            sub-requests.
        gwp_source: IPCC Assessment Report for GWP values.
        include_td_losses: Whether to include T&D loss emissions
            for electricity consumption.
        compliance_frameworks: Optional list of regulatory framework
            identifiers (e.g. 'GHG_PROTOCOL', 'CSRD', 'CDP') to
            run compliance checks against.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation request identifier",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Owning tenant identifier",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the target facility",
    )
    energy_requests: List[ElectricityConsumptionRequest] = Field(
        default_factory=list,
        description="List of electricity consumption sub-requests",
    )
    steam_heat_cool_requests: List[SteamHeatCoolingRequest] = Field(
        default_factory=list,
        description="List of steam/heat/cooling sub-requests",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    include_td_losses: bool = Field(
        default=True,
        description="Whether to include T&D loss emissions",
    )
    compliance_frameworks: Optional[List[str]] = Field(
        default=None,
        description="Regulatory frameworks for compliance checks",
    )

class GasEmissionDetail(GreenLangBase):
    """Breakdown of emissions for a single greenhouse gas species.

    Provides the individual gas emission quantity, the GWP multiplier
    used, and the resulting CO2-equivalent value.

    Attributes:
        gas: Greenhouse gas species.
        emission_kg: Direct emission quantity in kilograms.
        gwp_factor: GWP multiplier applied for CO2e conversion.
        co2e_kg: CO2-equivalent emission in kilograms.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species",
    )
    emission_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Direct emission quantity in kg",
    )
    gwp_factor: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="GWP multiplier for CO2e conversion",
    )
    co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2-equivalent emission in kg",
    )

class CalculationResult(GreenLangBase):
    """Result of a single Scope 2 location-based emission calculation.

    Contains the calculated emission quantities, emission factor
    metadata, gas-by-gas breakdown, provenance hash for audit
    trail, and processing metadata.

    Attributes:
        calculation_id: Unique identifier linking to the request.
        facility_id: Reference to the facility.
        energy_type: Type of purchased energy calculated.
        consumption_value: Consumption quantity used in the calculation.
        consumption_unit: Unit of the consumption quantity.
        grid_region: Grid region used for emission factor lookup.
        emission_factor_source: Source authority for the emission factor.
        ef_co2e_per_mwh: Total CO2e emission factor applied (kg/MWh).
        td_loss_pct: T&D loss percentage applied (0.0 if excluded).
        gas_breakdown: List of per-gas emission details.
        total_co2e_kg: Total CO2-equivalent emissions in kilograms.
        total_co2e_tonnes: Total CO2-equivalent emissions in tonnes.
        provenance_hash: SHA-256 hash of all calculation inputs and
            outputs for complete audit trail.
        calculated_at: UTC timestamp of the calculation.
        metadata: Optional additional metadata dictionary.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier linking to the request",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the facility",
    )
    energy_type: EnergyType = Field(
        ...,
        description="Type of purchased energy calculated",
    )
    consumption_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Consumption quantity used in the calculation",
    )
    consumption_unit: EnergyUnit = Field(
        ...,
        description="Unit of the consumption quantity",
    )
    grid_region: str = Field(
        ...,
        min_length=1,
        description="Grid region used for EF lookup",
    )
    emission_factor_source: EmissionFactorSource = Field(
        ...,
        description="Source authority for the emission factor",
    )
    ef_co2e_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2e emission factor (kg/MWh)",
    )
    td_loss_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="T&D loss percentage applied",
    )
    gas_breakdown: List[GasEmissionDetail] = Field(
        default_factory=list,
        description="Per-gas emission breakdown",
    )
    total_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2-equivalent emissions in kg",
    )
    total_co2e_tonnes: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2-equivalent emissions in tonnes",
    )
    provenance_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the calculation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional metadata",
    )

class BatchCalculationRequest(GreenLangBase):
    """Batch request for multiple Scope 2 location-based calculations.

    Aggregates multiple CalculationRequest instances for parallel
    processing across a portfolio of facilities.

    Attributes:
        batch_id: Unique identifier for this batch request.
        tenant_id: Owning tenant identifier for multi-tenancy.
        requests: List of individual calculation requests.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch request identifier",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Owning tenant identifier",
    )
    requests: List[CalculationRequest] = Field(
        ...,
        min_length=1,
        description="List of individual calculation requests",
    )

    @field_validator("requests")
    @classmethod
    def _validate_batch_size(
        cls, v: List[CalculationRequest],
    ) -> List[CalculationRequest]:
        """Validate that batch size does not exceed maximum."""
        if len(v) > MAX_CALCULATIONS_PER_BATCH:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum "
                f"{MAX_CALCULATIONS_PER_BATCH}"
            )
        return v

class BatchCalculationResult(GreenLangBase):
    """Result of a batch Scope 2 location-based calculation.

    Aggregates results from all individual calculations in a batch
    with portfolio-level totals.

    Attributes:
        batch_id: Unique identifier linking to the batch request.
        results: List of individual calculation results.
        total_co2e_tonnes: Portfolio-level total CO2e in tonnes.
        facility_count: Number of unique facilities in the batch.
        calculation_count: Total number of individual calculations.
        provenance_hash: SHA-256 hash of the complete batch for
            audit trail.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    batch_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier linking to the batch request",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    total_co2e_tonnes: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Portfolio-level total CO2e in tonnes",
    )
    facility_count: int = Field(
        ...,
        ge=0,
        description="Number of unique facilities in the batch",
    )
    calculation_count: int = Field(
        ...,
        ge=0,
        description="Total number of individual calculations",
    )
    provenance_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 provenance hash for batch audit trail",
    )

class ComplianceCheckResult(GreenLangBase):
    """Result of a regulatory compliance check for a calculation.

    Evaluates a completed calculation against a specific regulatory
    framework and reports findings and recommendations.

    Attributes:
        check_id: Unique identifier for this compliance check.
        calculation_id: Reference to the calculation being checked.
        framework: Regulatory framework identifier (e.g.
            'GHG_PROTOCOL', 'CSRD', 'CDP', 'ISO_14064').
        status: Overall compliance status.
        findings: List of specific compliance findings, each
            describing a requirement and its assessment.
        recommendations: List of recommended actions to achieve
            or maintain compliance.
        checked_at: UTC timestamp of the compliance check.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    check_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique compliance check identifier",
    )
    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the calculation being checked",
    )
    framework: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Regulatory framework identifier",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Specific compliance findings",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions for compliance",
    )
    checked_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the compliance check",
    )

class UncertaintyRequest(GreenLangBase):
    """Request for uncertainty quantification on a calculation result.

    Specifies the calculation to analyse and the uncertainty method
    parameters (Monte Carlo simulation or analytical propagation).

    Attributes:
        calculation_id: Reference to the calculation to analyse.
        method: Uncertainty quantification method ('monte_carlo'
            or 'analytical').
        iterations: Number of Monte Carlo iterations (applicable
            only when method is 'monte_carlo').
        confidence_level: Confidence level for the uncertainty
            interval (e.g. 0.95 for 95% CI).
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the calculation to analyse",
    )
    method: str = Field(
        default="monte_carlo",
        description="Uncertainty method (monte_carlo or analytical)",
    )
    iterations: int = Field(
        default=DEFAULT_MONTE_CARLO_ITERATIONS,
        ge=100,
        le=1_000_000,
        description="Number of Monte Carlo iterations",
    )
    confidence_level: Decimal = Field(
        default=DEFAULT_CONFIDENCE_LEVEL,
        ge=Decimal("0.50"),
        le=Decimal("0.9999"),
        description="Confidence level for the uncertainty interval",
    )

    @field_validator("method")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        """Validate that method is one of the allowed values."""
        allowed = {"monte_carlo", "analytical"}
        if v not in allowed:
            raise ValueError(
                f"method must be one of {allowed}, got '{v}'"
            )
        return v

class UncertaintyResult(GreenLangBase):
    """Result of uncertainty quantification for a calculation.

    Provides the mean, standard deviation, and confidence interval
    for the CO2e emission estimate.

    Attributes:
        calculation_id: Reference to the analysed calculation.
        method: Uncertainty method used.
        mean_co2e: Mean CO2e estimate in tonnes.
        std_dev: Standard deviation of the CO2e estimate in tonnes.
        ci_lower: Lower bound of the confidence interval in tonnes.
        ci_upper: Upper bound of the confidence interval in tonnes.
        confidence_level: Confidence level of the interval.
        iterations: Number of Monte Carlo iterations performed
            (0 for analytical method).
        distribution_params: Optional distribution parameters
            (e.g. shape, scale for lognormal).
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the analysed calculation",
    )
    method: str = Field(
        ...,
        description="Uncertainty method used",
    )
    mean_co2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Mean CO2e estimate in tonnes",
    )
    std_dev: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Standard deviation in tonnes",
    )
    ci_lower: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Lower bound of confidence interval in tonnes",
    )
    ci_upper: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Upper bound of confidence interval in tonnes",
    )
    confidence_level: Decimal = Field(
        ...,
        ge=Decimal("0.50"),
        le=Decimal("0.9999"),
        description="Confidence level of the interval",
    )
    iterations: int = Field(
        default=0,
        ge=0,
        description="Number of Monte Carlo iterations (0 for analytical)",
    )
    distribution_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional distribution parameters",
    )

class AggregationResult(GreenLangBase):
    """Aggregated emission result across multiple calculations.

    Provides portfolio-level or group-level totals for Scope 2
    location-based emissions, grouped by a specified dimension
    (facility, energy type, grid region, or time period).

    Attributes:
        aggregation_id: Unique identifier for this aggregation.
        group_by: Dimension used for grouping (e.g. 'facility',
            'energy_type', 'grid_region', 'month').
        period: Reporting period description (e.g. '2025',
            '2025-Q1', '2025-01').
        total_co2e_tonnes: Aggregated total CO2e in tonnes.
        facility_count: Number of unique facilities in the group.
        details: List of per-group detail records, each a dictionary
            with the group key and subtotal.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    aggregation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique aggregation identifier",
    )
    group_by: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Dimension used for grouping",
    )
    period: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reporting period description",
    )
    total_co2e_tonnes: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Aggregated total CO2e in tonnes",
    )
    facility_count: int = Field(
        ...,
        ge=0,
        description="Number of unique facilities in the group",
    )
    details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-group detail records",
    )

class GridFactorLookupResult(GreenLangBase):
    """Result of a grid emission factor lookup operation.

    Returned by the grid factor database engine when resolving a
    facility's grid region to an emission factor set.

    Attributes:
        region_id: Grid region identifier that was resolved.
        source: Authoritative source of the emission factor.
        year: Reporting year of the emission factor.
        co2_kg_per_mwh: CO2 emission rate in kg per MWh.
        ch4_kg_per_mwh: CH4 emission rate in kg per MWh.
        n2o_kg_per_mwh: N2O emission rate in kg per MWh.
        total_co2e_kg_per_mwh: Total CO2e rate in kg per MWh.
        td_loss_pct: Associated T&D loss percentage for the region.
        quality_tier: Data quality tier of the resolved factor.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    region_id: str = Field(
        ...,
        min_length=1,
        description="Grid region identifier",
    )
    source: EmissionFactorSource = Field(
        ...,
        description="Authoritative source of the emission factor",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year of the emission factor",
    )
    co2_kg_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2 emission rate in kg per MWh",
    )
    ch4_kg_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CH4 emission rate in kg per MWh",
    )
    n2o_kg_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="N2O emission rate in kg per MWh",
    )
    total_co2e_kg_per_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2e rate in kg per MWh",
    )
    td_loss_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="T&D loss percentage for the region",
    )
    quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier of the resolved factor",
    )

class TDLossResult(GreenLangBase):
    """Result of a transmission and distribution loss calculation.

    Provides the resolved T&D loss percentage, gross and net
    consumption values, and the emission quantity attributable
    to grid losses.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        td_loss_pct: Resolved T&D loss percentage as a decimal
            fraction (e.g. 0.05 for 5%).
        method: Method used to determine the T&D loss factor.
        gross_consumption: Gross consumption including T&D losses
            (in MWh or GJ depending on energy type).
        net_consumption: Net metered consumption (in MWh or GJ).
        loss_emissions_kg: CO2e emissions attributable to T&D
            losses in kilograms.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    td_loss_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="T&D loss percentage (0.0 - 1.0)",
    )
    method: TDLossMethod = Field(
        ...,
        description="Method used to determine the T&D loss factor",
    )
    gross_consumption: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Gross consumption including T&D losses",
    )
    net_consumption: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Net metered consumption",
    )
    loss_emissions_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2e emissions from T&D losses in kg",
    )
