# -*- coding: utf-8 -*-
"""
Scope 2 Market-Based Emissions Agent Data Models - AGENT-MRV-010

Pydantic v2 data models for the Scope 2 Market-Based Emissions Agent SDK
covering GHG Protocol Scope 2 market-based electricity, steam, heating,
and cooling emission calculations including:
- 10 contractual instrument types (PPA, REC, GO, REGO, I-REC, T-REC,
  J-Credit, LGC, Green Tariff, Supplier-Specific)
- 7 GHG Protocol Scope 2 quality criteria for contractual instruments
- 8 tracking system registries (Green-e, AIB EECS, Ofgem, I-REC Standard,
  M-RETS, NAR, WREGIS, Custom)
- 65+ residual mix emission factors by region (US subregions, EU countries,
  APAC, Americas, Global)
- 11 energy source emission factors for instrument-specific calculations
- Instrument quality assessment and validation framework
- Supplier-specific emission factor management
- Allocation of contractual instruments to energy purchases
- Dual reporting (location-based vs. market-based comparison)
- Coverage analysis (fully covered, partially covered, uncovered)
- GWP values from IPCC AR4, AR5, AR6, and AR6 20-year horizon
- Unit conversions between kWh, MWh, GJ, MMBtu, and therms
- Facility metadata with grid region mapping
- Batch calculation requests across multiple facilities
- Monte Carlo and analytical uncertainty quantification
- Multi-framework regulatory compliance checking
- Aggregation by facility, instrument type, energy source, or time period
- SHA-256 provenance chain for complete audit trails

Enumerations (20):
    - InstrumentType, InstrumentStatus, EnergySource, EnergyType,
      EnergyUnit, CalculationMethod, EmissionGas, GWPSource,
      QualityCriterion, TrackingSystem, ResidualMixSource, FacilityType,
      ComplianceStatus, CoverageStatus, ReportingPeriod, ContractType,
      DataQualityTier, DualReportingStatus, AllocationMethod,
      ConsumptionDataSource

Constants (all Decimal for zero-hallucination deterministic arithmetic):
    - GWP_VALUES: IPCC AR4/AR5/AR6/AR6_20YR GWP (CO2, CH4, N2O)
    - RESIDUAL_MIX_FACTORS: 65+ regional residual mix EFs (kgCO2e/kWh)
    - ENERGY_SOURCE_EF: 11 energy source emission factors (kgCO2e/kWh)
    - SUPPLIER_DEFAULT_EF: Default supplier EFs by country (kgCO2e/kWh)
    - INSTRUMENT_QUALITY_WEIGHTS: Quality criterion weights
    - VINTAGE_VALIDITY_YEARS: Maximum vintage age by instrument type
    - UNIT_CONVERSIONS: Energy unit conversion factors

Data Models (20):
    - ContractualInstrument, InstrumentQualityAssessment,
      SupplierEmissionFactor, ResidualMixFactor, EnergyPurchase,
      FacilityInfo, AllocationResult, CoveredEmissionResult,
      UncoveredEmissionResult, GasEmissionDetail, MarketBasedResult,
      DualReportingResult, CalculationRequest, BatchCalculationRequest,
      BatchCalculationResult, ComplianceCheckResult,
      InstrumentValidationResult, UncertaintyRequest, UncertaintyResult,
      AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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

#: Maximum number of energy purchase records per calculation request.
MAX_ENERGY_PURCHASES_PER_CALC: int = 1_000

#: Maximum number of instruments per energy purchase record.
MAX_INSTRUMENTS_PER_PURCHASE: int = 100

#: Default Monte Carlo simulation iterations for uncertainty analysis.
DEFAULT_MONTE_CARLO_ITERATIONS: int = 10_000

#: Default confidence level for uncertainty intervals.
DEFAULT_CONFIDENCE_LEVEL: Decimal = Decimal("0.95")

#: Default quality threshold for instrument validation (0-1 scale).
DEFAULT_QUALITY_THRESHOLD: Decimal = Decimal("0.70")

#: Prefix for all database table names in this module.
TABLE_PREFIX: str = "gl_s2m_"


# =============================================================================
# Enumerations (20)
# =============================================================================


class InstrumentType(str, Enum):
    """Classification of contractual instruments for market-based Scope 2.

    GHG Protocol Scope 2 Guidance defines a hierarchy of contractual
    instruments that convey emission factor information from the
    generator to the consumer. Each instrument type has different
    implications for emission factor quality, geographic applicability,
    and regulatory acceptance.

    PPA: Power Purchase Agreement. A bilateral contract between a
        generator and an off-taker specifying the price, quantity,
        and delivery terms for electricity. Physical PPAs involve
        direct electricity delivery; virtual PPAs are financial
        instruments with separate EAC retirement.
    REC: Renewable Energy Certificate (US). A tradeable certificate
        representing the environmental attributes of 1 MWh of
        renewable electricity generation. Tracked in US registries
        (M-RETS, NAR, WREGIS).
    GO: Guarantee of Origin (EU). An electronic certificate proving
        that 1 MWh of electricity was generated from a specified
        energy source. Issued under EU Directive 2018/2001/EC and
        tracked by AIB EECS.
    REGO: Renewable Energy Guarantee of Origin (UK). The UK equivalent
        of the EU GO, issued by Ofgem for 1 MWh of renewable
        electricity generated in the UK.
    I_REC: International Renewable Energy Certificate. A global EAC
        standard administered by the I-REC Standard Foundation,
        used in markets without local EAC systems.
    T_REC: Tradeable Renewable Energy Certificate (various Asian
        markets). Similar to I-REC but issued under national
        programmes in specific Asian countries.
    J_CREDIT: J-Credit (Japan). A carbon offset credit issued under
        Japan's Joint Crediting Mechanism. Applicable for Japanese
        Scope 2 market-based reporting.
    LGC: Large-scale Generation Certificate (Australia). An EAC
        issued under the Australian Renewable Energy Target (RET)
        scheme for each MWh of eligible renewable generation.
    GREEN_TARIFF: Green electricity tariff from a retail supplier.
        The supplier procures and retires EACs on behalf of the
        consumer, backed by a contractual commitment.
    SUPPLIER_SPECIFIC: Supplier-specific emission factor from a
        utility or energy retailer disclosure, typically based on
        the supplier's generation mix or procurement portfolio.
    """

    PPA = "ppa"
    REC = "rec"
    GO = "go"
    REGO = "rego"
    I_REC = "i_rec"
    T_REC = "t_rec"
    J_CREDIT = "j_credit"
    LGC = "lgc"
    GREEN_TARIFF = "green_tariff"
    SUPPLIER_SPECIFIC = "supplier_specific"


class InstrumentStatus(str, Enum):
    """Lifecycle status of a contractual instrument.

    Tracks the current disposition of a contractual instrument to
    prevent double counting and ensure only valid instruments are
    applied to market-based emission calculations.

    ACTIVE: Instrument is valid and available for allocation to
        energy purchases. Has not been retired, cancelled, or
        expired.
    RETIRED: Instrument has been permanently retired (consumed)
        against a specific energy purchase or reporting period.
        Cannot be re-used or transferred. Retirement is recorded
        in the tracking system registry.
    EXPIRED: Instrument has exceeded its vintage validity period
        (typically 1-5 years depending on instrument type and
        jurisdiction). Cannot be used for current-year reporting.
    CANCELLED: Instrument has been voluntarily cancelled or
        invalidated by the issuing authority. Cannot be used for
        any reporting purpose.
    PENDING: Instrument is in the process of being issued, verified,
        or transferred. Not yet available for allocation.
    """

    ACTIVE = "active"
    RETIRED = "retired"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PENDING = "pending"


class EnergySource(str, Enum):
    """Classification of electricity generation source for instruments.

    Identifies the primary energy source associated with a contractual
    instrument. Determines the emission factor applied when the
    instrument is allocated to an energy purchase for market-based
    calculations.

    SOLAR: Photovoltaic or concentrated solar power generation.
        Zero direct emissions. EF = 0 kgCO2e/kWh.
    WIND: Onshore or offshore wind turbine generation. Zero direct
        emissions. EF = 0 kgCO2e/kWh.
    HYDRO: Conventional hydroelectric or run-of-river generation.
        Zero direct emissions (biogenic reservoir emissions
        excluded per GHG Protocol). EF = 0 kgCO2e/kWh.
    NUCLEAR: Nuclear fission power generation. Zero direct CO2
        emissions at the point of generation. EF = 0 kgCO2e/kWh.
    BIOMASS: Biomass or biogas combustion generation. Biogenic CO2
        reported separately; fossil CO2e = 0 for sustainably
        sourced biomass per GHG Protocol guidance.
    GEOTHERMAL: Geothermal power generation. Low direct emissions
        from dissolved gases in geothermal fluids.
    NATURAL_GAS_CCGT: Combined cycle gas turbine natural gas
        generation. Lower emission intensity than simple cycle.
    NATURAL_GAS_OCGT: Open cycle gas turbine natural gas generation.
        Higher emission intensity than combined cycle.
    COAL: Coal-fired power generation. Highest emission intensity
        among fossil fuel sources.
    OIL: Oil-fired power generation. High emission intensity,
        common in island and remote grids.
    MIXED: Mixed or unspecified generation source. Used when the
        instrument does not specify a single generation technology.
    """

    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    NUCLEAR = "nuclear"
    BIOMASS = "biomass"
    GEOTHERMAL = "geothermal"
    NATURAL_GAS_CCGT = "natural_gas_ccgt"
    NATURAL_GAS_OCGT = "natural_gas_ocgt"
    COAL = "coal"
    OIL = "oil"
    MIXED = "mixed"


class EnergyType(str, Enum):
    """Classification of purchased energy types for Scope 2 reporting.

    GHG Protocol Scope 2 Guidance requires organisations to report
    indirect emissions from four categories of purchased or acquired
    energy. Market-based calculations apply contractual instruments
    and supplier-specific emission factors to these energy types.

    ELECTRICITY: Grid-supplied electrical energy. Market-based
        emissions use contractual instrument emission factors or
        residual mix factors when no instruments are allocated.
    STEAM: Purchased steam from external suppliers or district
        steam networks. Emission factors may come from supplier
        disclosure or default factors.
    HEATING: Purchased heat from district heating networks or
        external boiler systems. Factors may come from supplier
        disclosure or default factors.
    COOLING: Purchased chilled water or cooling from district
        cooling networks. Factors may come from supplier disclosure
        or default factors.
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
        Standard basis for electricity emission factors and
        contractual instrument quantities.
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


class CalculationMethod(str, Enum):
    """Methodology for calculating Scope 2 market-based emissions.

    GHG Protocol Scope 2 Guidance defines a hierarchy of methods
    for market-based reporting. Organisations should apply the
    highest-quality method for which they have valid data.

    INSTRUMENT_BASED: Uses emission factors from contractual
        instruments (EACs, PPAs, green tariffs) retired against
        the reporting entity's energy consumption. Highest quality
        when instruments meet all GHG Protocol quality criteria.
        Applied only to the portion of consumption covered by
        valid instruments.
    SUPPLIER_SPECIFIC: Uses emission factors disclosed by the
        energy supplier based on their generation mix or
        procurement portfolio. Requires supplier attestation
        that the factors have not been double-counted.
    RESIDUAL_MIX: Uses the residual mix emission factor for the
        grid region, which represents the emission intensity of
        generation not claimed by any entity through contractual
        instruments. Applied to any consumption not covered by
        instruments or supplier-specific factors.
    HYBRID: Combines multiple methods: instrument-based for
        covered consumption, supplier-specific where available,
        and residual mix for the remainder. The standard approach
        for organisations with partial EAC coverage.
    """

    INSTRUMENT_BASED = "instrument_based"
    SUPPLIER_SPECIFIC = "supplier_specific"
    RESIDUAL_MIX = "residual_mix"
    HYBRID = "hybrid"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in Scope 2 market-based calculations.

    CO2: Carbon dioxide. Primary emission from fossil-fuel-based
        electricity generation. For renewable instruments, the CO2
        emission factor is zero (generation-only; lifecycle
        emissions are excluded).
    CH4: Methane. Emitted from upstream fuel extraction and
        incomplete combustion at power plants. Typically zero for
        renewable instruments. Small fraction of total Scope 2.
    N2O: Nitrous oxide. Emitted from combustion processes at power
        plants. Typically zero for renewable instruments. Included
        for full GHG Protocol compliance.
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


class QualityCriterion(str, Enum):
    """GHG Protocol Scope 2 quality criteria for contractual instruments.

    The GHG Protocol Scope 2 Guidance defines seven quality criteria
    that contractual instruments should meet to be considered valid
    for market-based emission reporting. Instruments are scored
    against each criterion.

    UNIQUE_CLAIM: The instrument conveys a unique emission claim
        that has not been and will not be used by any other entity.
        Ensures single-ownership of environmental attributes.
    ASSOCIATED_DELIVERY: The instrument is associated with the
        delivery of electricity to the reporting entity's grid or
        the same market. Physical delivery or same-market
        criterion.
    TEMPORAL_MATCH: The instrument's vintage year matches the
        reporting period. Generation and consumption occur within
        the same reporting year or allowable vintage window.
    GEOGRAPHIC_MATCH: The instrument was generated within the same
        grid, market, or country as the reporting entity's
        electricity consumption. Ensures market relevance.
    NO_DOUBLE_COUNT: The instrument has not been counted towards
        another entity's emission inventory or compliance
        obligation. Anti-double-counting provision.
    RECOGNIZED_REGISTRY: The instrument is tracked and retired in
        a recognised tracking system (e.g. Green-e, AIB EECS,
        Ofgem, I-REC Standard). Ensures verifiable retirement.
    REPRESENTS_GENERATION: The instrument represents actual
        electricity generation (not offsets or avoided emissions).
        Ensures the instrument conveys generation attributes.
    """

    UNIQUE_CLAIM = "unique_claim"
    ASSOCIATED_DELIVERY = "associated_delivery"
    TEMPORAL_MATCH = "temporal_match"
    GEOGRAPHIC_MATCH = "geographic_match"
    NO_DOUBLE_COUNT = "no_double_count"
    RECOGNIZED_REGISTRY = "recognized_registry"
    REPRESENTS_GENERATION = "represents_generation"


class TrackingSystem(str, Enum):
    """Recognised tracking system registries for contractual instruments.

    Tracking systems provide the infrastructure for issuing, tracking,
    transferring, and retiring energy attribute certificates. Each
    system operates within a specific geographic domain.

    GREEN_E: Green-e Energy (Center for Resource Solutions). US
        voluntary market. Certifies RECs and green power products.
        Requires annual verification and consumer disclosure.
    AIB_EECS: Association of Issuing Bodies European Energy
        Certificate System. Pan-European GO tracking. Standardised
        across 20+ European countries. Domain protocol ensures
        no double issuance or counting.
    OFGEM: UK Office of Gas and Electricity Markets. Issues and
        tracks REGOs for UK renewable electricity generation.
    I_REC_STANDARD: I-REC Standard Foundation. Global EAC tracking
        for markets without local systems. Operates in 50+ countries
        across Asia, Africa, Latin America, and Middle East.
    M_RETS: Midwest Renewable Energy Tracking System. US regional
        registry serving the central US and parts of Canada.
    NAR: North American Renewables Registry. US national registry
        providing REC tracking across multiple state compliance
        and voluntary markets.
    WREGIS: Western Renewable Energy Generation Information System.
        US western interconnection registry tracking RECs from
        generators in 15 western US states.
    CUSTOM: User-defined or proprietary tracking system for
        instruments not tracked in a recognised public registry.
        Requires additional evidence documentation.
    """

    GREEN_E = "green_e"
    AIB_EECS = "aib_eecs"
    OFGEM = "ofgem"
    I_REC_STANDARD = "i_rec_standard"
    M_RETS = "m_rets"
    NAR = "nar"
    WREGIS = "wregis"
    CUSTOM = "custom"


class ResidualMixSource(str, Enum):
    """Source authority for residual mix emission factors.

    Residual mix factors represent the emission intensity of
    electricity generation that has not been claimed by any entity
    through contractual instruments. Applied to uncovered
    consumption in market-based calculations.

    AIB: Association of Issuing Bodies European Residual Mix.
        Published annually for EU/EEA member states. Calculated
        by removing all tracked GO volumes from the national
        generation mix.
    GREEN_E: Green-e Residual Mix. US residual mix factors
        calculated by removing tracked REC volumes from regional
        generation mixes (eGRID subregion basis).
    NATIONAL: National residual mix published by a country's
        competent authority or grid operator.
    ESTIMATED: Estimated residual mix derived from national
        generation mix adjusted for known EAC volumes. Used
        when no official residual mix is published.
    CUSTOM: User-provided residual mix factor with documented
        evidence and methodology.
    """

    AIB = "aib"
    GREEN_E = "green_e"
    NATIONAL = "national"
    ESTIMATED = "estimated"
    CUSTOM = "custom"


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


class CoverageStatus(str, Enum):
    """Coverage status of energy consumption by contractual instruments.

    Classifies the degree to which a facility's or purchase's energy
    consumption is covered by valid contractual instruments for
    market-based emission reporting.

    FULLY_COVERED: 100% of consumption is covered by retired
        contractual instruments. All emissions calculated using
        instrument emission factors.
    PARTIALLY_COVERED: Some but not all consumption is covered by
        instruments. Covered portion uses instrument EFs; uncovered
        portion uses residual mix EF.
    UNCOVERED: No contractual instruments are allocated. All
        emissions calculated using the residual mix emission factor
        for the region.
    OVER_COVERED: Instrument quantity exceeds consumption quantity.
        Excess instruments are not applied. A warning is generated
        to flag potential instrument management issues.
    """

    FULLY_COVERED = "fully_covered"
    PARTIALLY_COVERED = "partially_covered"
    UNCOVERED = "uncovered"
    OVER_COVERED = "over_covered"


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


class ContractType(str, Enum):
    """Classification of power purchase agreement contract structures.

    Determines how the contractual instrument is linked to the
    physical delivery of electricity and influences the geographic
    match quality criterion assessment.

    PHYSICAL_PPA: Physical power purchase agreement with direct
        electricity delivery from the generator to the off-taker
        through the grid. Highest geographic match quality.
    VIRTUAL_PPA: Virtual (financial) power purchase agreement
        structured as a contract for differences (CfD). No physical
        delivery; EACs are retired separately. Geographic match
        depends on market alignment.
    SLEEVED_PPA: Sleeved PPA where a third-party utility acts as
        intermediary between the generator and the off-taker.
        Physical delivery through the utility's retail supply.
    DIRECT_PURCHASE: Direct purchase of EACs (RECs, GOs, I-RECs)
        from a broker, exchange, or generator without an associated
        PPA. Lowest geographic match if purchased from a different
        market.
    """

    PHYSICAL_PPA = "physical_ppa"
    VIRTUAL_PPA = "virtual_ppa"
    SLEEVED_PPA = "sleeved_ppa"
    DIRECT_PURCHASE = "direct_purchase"


class DataQualityTier(str, Enum):
    """Data quality classification for emission factor inputs.

    Tier classification follows the IPCC approach to data quality
    and determines the uncertainty range applied to calculation
    results.

    TIER_1: Default values from international databases (IPCC, IEA).
        Highest uncertainty range (+/- 25-50%). Used when no
        instrument-specific or supplier-specific data is available.
    TIER_2: Supplier-specific or instrument-based emission factors
        from recognised tracking systems. Moderate uncertainty
        (+/- 10-25%).
    TIER_3: Instrument-specific emission factors from verified
        PPAs with direct metering and independent audit.
        Lowest uncertainty (+/- 5-10%).
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class DualReportingStatus(str, Enum):
    """Completeness status of dual reporting (location + market-based).

    GHG Protocol Scope 2 Guidance requires organisations to report
    both location-based and market-based Scope 2 totals. This enum
    tracks whether both methods have been calculated.

    COMPLETE: Both location-based and market-based calculations
        have been completed for the reporting entity and period.
    LOCATION_ONLY: Only the location-based calculation has been
        completed. Market-based calculation is pending or not
        yet available.
    MARKET_ONLY: Only the market-based calculation has been
        completed. Location-based calculation is pending or not
        yet available.
    """

    COMPLETE = "complete"
    LOCATION_ONLY = "location_only"
    MARKET_ONLY = "market_only"


class AllocationMethod(str, Enum):
    """Method for allocating contractual instruments to energy purchases.

    Determines the order and logic for matching available instruments
    to energy consumption when multiple instruments are available.

    PRIORITY_BASED: Allocates instruments in priority order based
        on instrument type hierarchy (PPA > green tariff > REC/GO >
        supplier-specific). Higher-quality instruments are applied
        first.
    PROPORTIONAL: Allocates instruments proportionally based on
        their quantity relative to total available instrument
        volume. Spreads allocation evenly across all instruments.
    CUSTOM: User-defined allocation rules specified in the
        calculation request. Supports explicit instrument-to-
        purchase mapping.
    """

    PRIORITY_BASED = "priority_based"
    PROPORTIONAL = "proportional"
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
# Residual Mix Emission Factors
# ---------------------------------------------------------------------------

#: Regional residual mix emission factors for market-based Scope 2.
#: Units: kgCO2e per kWh.
#: Source: Green-e (US), AIB (EU), national registries (APAC/Americas),
#:         and estimated factors for regions without official residual mix.
#:
#: Residual mix represents the emission intensity of electricity
#: generation not claimed by any entity through contractual instruments.
#: Applied to uncovered consumption in market-based calculations.
#:
#: Keys follow the pattern: {REGION}-{SUBREGION} for sub-national
#: regions, or GLOBAL for the world average fallback.
#:
#: US subregions are based on eGRID subregion boundaries with
#: adjustments for tracked REC volumes. EU factors are from the
#: AIB European Residual Mix published annually. APAC and Americas
#: factors are estimated from national generation mixes adjusted
#: for known EAC retirement volumes.
RESIDUAL_MIX_FACTORS: Dict[str, Decimal] = {
    # ---- US Subregions (Green-e residual mix, eGRID-aligned) ----
    "US-CAMX": Decimal("0.285"),
    "US-ERCT": Decimal("0.420"),
    "US-NEWE": Decimal("0.295"),
    "US-NWPP": Decimal("0.355"),
    "US-RFCE": Decimal("0.385"),
    "US-RFCW": Decimal("0.440"),
    "US-MROE": Decimal("0.545"),
    "US-MROW": Decimal("0.470"),
    "US-SRMV": Decimal("0.410"),
    "US-SRSO": Decimal("0.480"),
    "US-SRVC": Decimal("0.365"),
    "US-SRTV": Decimal("0.415"),
    "US-SPNO": Decimal("0.465"),
    "US-SPSO": Decimal("0.490"),
    "US-RMPA": Decimal("0.510"),
    "US-NYCW": Decimal("0.280"),
    "US-NYLI": Decimal("0.390"),
    "US-NYUP": Decimal("0.210"),
    "US-AVG": Decimal("0.425"),
    # ---- EU Countries (AIB European Residual Mix) ----
    "EU-DE": Decimal("0.520"),
    "EU-FR": Decimal("0.085"),
    "EU-GB": Decimal("0.285"),
    "EU-ES": Decimal("0.245"),
    "EU-IT": Decimal("0.405"),
    "EU-NL": Decimal("0.485"),
    "EU-PL": Decimal("0.745"),
    "EU-SE": Decimal("0.045"),
    "EU-NO": Decimal("0.025"),
    "EU-DK": Decimal("0.210"),
    "EU-FI": Decimal("0.110"),
    "EU-AT": Decimal("0.185"),
    "EU-BE": Decimal("0.230"),
    "EU-PT": Decimal("0.255"),
    "EU-IE": Decimal("0.380"),
    "EU-CZ": Decimal("0.530"),
    "EU-RO": Decimal("0.310"),
    "EU-HU": Decimal("0.275"),
    "EU-GR": Decimal("0.480"),
    "EU-BG": Decimal("0.465"),
    "EU-SK": Decimal("0.155"),
    "EU-HR": Decimal("0.215"),
    "EU-SI": Decimal("0.270"),
    "EU-LT": Decimal("0.145"),
    "EU-LV": Decimal("0.130"),
    "EU-EE": Decimal("0.625"),
    "EU-AVG": Decimal("0.380"),
    # ---- APAC (National / Estimated Residual Mix) ----
    "APAC-AU": Decimal("0.750"),
    "APAC-JP": Decimal("0.520"),
    "APAC-SG": Decimal("0.425"),
    "APAC-KR": Decimal("0.495"),
    "APAC-IN": Decimal("0.780"),
    "APAC-CN": Decimal("0.600"),
    # ---- Americas (National / Estimated Residual Mix) ----
    "AMER-CA": Decimal("0.145"),
    "AMER-MX": Decimal("0.480"),
    "AMER-BR": Decimal("0.095"),
    # ---- Global Fallback ----
    "GLOBAL": Decimal("0.500"),
}


# ---------------------------------------------------------------------------
# Energy Source Emission Factors
# ---------------------------------------------------------------------------

#: Emission factors by electricity generation source.
#: Units: kgCO2e per kWh of generation.
#: Source: IPCC 2006 Guidelines, lifecycle-adjusted for direct emissions
#:         at the point of generation. Renewable sources have zero
#:         direct emission factors per GHG Protocol Scope 2 convention.
#:
#: These factors are applied when calculating emissions from
#: contractual instruments with a known generation source.
#: For instruments backed by renewable generation, the emission
#: factor is zero, reflecting no direct combustion emissions.
ENERGY_SOURCE_EF: Dict[str, Decimal] = {
    "solar": Decimal("0.000"),
    "wind": Decimal("0.000"),
    "hydro": Decimal("0.000"),
    "nuclear": Decimal("0.000"),
    "biomass": Decimal("0.000"),
    "geothermal": Decimal("0.038"),
    "natural_gas_ccgt": Decimal("0.370"),
    "natural_gas_ocgt": Decimal("0.500"),
    "coal": Decimal("0.900"),
    "oil": Decimal("0.650"),
    "mixed": Decimal("0.450"),
}


# ---------------------------------------------------------------------------
# Supplier Default Emission Factors
# ---------------------------------------------------------------------------

#: Default supplier-specific emission factors by country.
#: Units: kgCO2e per kWh.
#: Source: National energy regulatory authority disclosures, utility
#:         average generation mix data, and IEA electricity factors.
#:
#: Used as a fallback when a supplier has not disclosed a specific
#: emission factor. These represent the national average utility
#: emission intensity and are less accurate than verified
#: supplier-specific disclosures.
#:
#: Keys use ISO 3166-1 alpha-2 country codes (uppercase).
SUPPLIER_DEFAULT_EF: Dict[str, Decimal] = {
    # Americas
    "US": Decimal("0.390"),
    "CA": Decimal("0.130"),
    "MX": Decimal("0.440"),
    "BR": Decimal("0.080"),
    # Europe
    "GB": Decimal("0.225"),
    "DE": Decimal("0.350"),
    "FR": Decimal("0.060"),
    "IT": Decimal("0.240"),
    "ES": Decimal("0.145"),
    "PL": Decimal("0.650"),
    "NL": Decimal("0.340"),
    "BE": Decimal("0.160"),
    "SE": Decimal("0.010"),
    "NO": Decimal("0.010"),
    "DK": Decimal("0.120"),
    "FI": Decimal("0.075"),
    "AT": Decimal("0.090"),
    "CH": Decimal("0.020"),
    "IE": Decimal("0.305"),
    "PT": Decimal("0.185"),
    "GR": Decimal("0.360"),
    "CZ": Decimal("0.405"),
    "RO": Decimal("0.275"),
    "HU": Decimal("0.225"),
    "BG": Decimal("0.385"),
    "SK": Decimal("0.110"),
    "HR": Decimal("0.165"),
    "SI": Decimal("0.220"),
    "LT": Decimal("0.040"),
    "LV": Decimal("0.105"),
    "EE": Decimal("0.590"),
    # Asia-Pacific
    "CN": Decimal("0.570"),
    "JP": Decimal("0.470"),
    "IN": Decimal("0.720"),
    "KR": Decimal("0.425"),
    "AU": Decimal("0.670"),
    "NZ": Decimal("0.090"),
    "SG": Decimal("0.415"),
    "TW": Decimal("0.510"),
    "TH": Decimal("0.440"),
    "MY": Decimal("0.560"),
    "ID": Decimal("0.720"),
    "PH": Decimal("0.555"),
    # Middle East
    "SA": Decimal("0.590"),
    "AE": Decimal("0.405"),
    "IL": Decimal("0.500"),
    # Africa
    "ZA": Decimal("0.935"),
    "EG": Decimal("0.430"),
    "NG": Decimal("0.415"),
    "KE": Decimal("0.100"),
    # Russia and Central Asia
    "RU": Decimal("0.350"),
    "TR": Decimal("0.395"),
}


# ---------------------------------------------------------------------------
# Instrument Quality Criterion Weights
# ---------------------------------------------------------------------------

#: Weight assigned to each GHG Protocol quality criterion when
#: calculating the overall quality score for a contractual instrument.
#: Units: dimensionless weight (sum = 1.0).
#: Source: GreenLang internal methodology aligned with GHG Protocol
#:         Scope 2 Guidance (2015) quality criteria hierarchy.
#:
#: Higher weights reflect criteria with greater impact on emission
#: factor reliability and anti-double-counting assurance.
INSTRUMENT_QUALITY_WEIGHTS: Dict[str, Decimal] = {
    "unique_claim": Decimal("0.20"),
    "associated_delivery": Decimal("0.15"),
    "temporal_match": Decimal("0.15"),
    "geographic_match": Decimal("0.15"),
    "no_double_count": Decimal("0.15"),
    "recognized_registry": Decimal("0.10"),
    "represents_generation": Decimal("0.10"),
}


# ---------------------------------------------------------------------------
# Vintage Validity Years
# ---------------------------------------------------------------------------

#: Maximum allowed vintage age (in years) for each instrument type.
#: An instrument whose vintage year is more than this many years
#: before the reporting period end year is considered expired.
#: Source: GHG Protocol Scope 2 Guidance, Green-e programme rules,
#:         AIB EECS rules, I-REC Standard rules.
#:
#: PPA instruments have longer validity because they represent
#: long-term contractual commitments with ongoing generation.
VINTAGE_VALIDITY_YEARS: Dict[str, int] = {
    "ppa": 5,
    "rec": 2,
    "go": 1,
    "rego": 1,
    "i_rec": 2,
    "t_rec": 2,
    "j_credit": 3,
    "lgc": 3,
    "green_tariff": 1,
    "supplier_specific": 1,
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
# Data Models (20)
# =============================================================================


class ContractualInstrument(BaseModel):
    """A contractual instrument conveying emission attributes for market-based.

    Represents a single contractual instrument (PPA, REC, GO, REGO,
    I-REC, T-REC, J-Credit, LGC, green tariff, or supplier-specific
    disclosure) that conveys the environmental attributes of electricity
    generation from a specific source to the reporting entity.

    Each instrument has a defined quantity (in MWh), an associated
    energy source and emission factor, a vintage year, and a lifecycle
    status. Instruments are allocated to energy purchases to calculate
    market-based Scope 2 emissions.

    Attributes:
        instrument_id: Unique system identifier for the instrument (UUID).
        instrument_type: Classification of the instrument type.
        quantity_mwh: Quantity of electricity represented by the
            instrument in megawatt-hours. Typically 1 MWh per
            certificate for RECs, GOs, and I-RECs.
        energy_source: Primary energy source of the generation
            facility that produced the instrument.
        ef_kgco2e_per_kwh: Emission factor associated with the
            instrument in kgCO2e per kWh. Zero for renewable
            instruments (solar, wind, hydro, nuclear, biomass).
        vintage_year: Year of electricity generation for which the
            instrument was issued. Must match or be within the
            allowable vintage window of the reporting period.
        tracking_system: Registry or tracking system in which the
            instrument is registered and will be retired.
        certificate_id: Unique certificate identifier within the
            tracking system (e.g. Green-e serial number, AIB
            certificate ID).
        status: Current lifecycle status of the instrument.
        region: Geographic region of the generation facility (e.g.
            US-CAMX, EU-DE, APAC-AU).
        verified: Whether the instrument has been independently
            verified or audited.
        contract_type: Type of contractual arrangement (PPA type
            or direct purchase).
        delivery_start: Start date of the electricity delivery
            period covered by the instrument.
        delivery_end: End date of the electricity delivery period
            covered by the instrument.
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    instrument_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique system identifier for the instrument (UUID)",
    )
    instrument_type: InstrumentType = Field(
        ...,
        description="Classification of the instrument type",
    )
    quantity_mwh: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Quantity of electricity in MWh",
    )
    energy_source: EnergySource = Field(
        ...,
        description="Primary energy source of the generation facility",
    )
    ef_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor in kgCO2e per kWh",
    )
    vintage_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Year of electricity generation",
    )
    tracking_system: TrackingSystem = Field(
        ...,
        description="Registry tracking system for the instrument",
    )
    certificate_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique certificate identifier in the tracking system",
    )
    status: InstrumentStatus = Field(
        default=InstrumentStatus.ACTIVE,
        description="Current lifecycle status of the instrument",
    )
    region: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Geographic region of the generation facility",
    )
    verified: bool = Field(
        default=False,
        description="Whether independently verified or audited",
    )
    contract_type: Optional[ContractType] = Field(
        default=None,
        description="Type of contractual arrangement (PPA or purchase)",
    )
    delivery_start: datetime = Field(
        ...,
        description="Start date of the delivery period",
    )
    delivery_end: datetime = Field(
        ...,
        description="End date of the delivery period",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Owning tenant identifier for multi-tenancy",
    )

    @field_validator("delivery_end")
    @classmethod
    def _delivery_end_after_start(cls, v: datetime, info: Any) -> datetime:
        """Validate that delivery_end is after delivery_start."""
        start = info.data.get("delivery_start")
        if start is not None and v <= start:
            raise ValueError("delivery_end must be after delivery_start")
        return v


class InstrumentQualityAssessment(BaseModel):
    """Quality assessment of a contractual instrument against GHG Protocol.

    Evaluates a contractual instrument against the seven GHG Protocol
    Scope 2 quality criteria and produces a weighted overall score.
    Instruments scoring below the quality threshold are flagged as
    not meeting minimum quality requirements.

    Attributes:
        instrument_id: Reference to the assessed instrument.
        unique_claim_score: Score for the unique claim criterion (0-1).
        associated_delivery_score: Score for the associated delivery
            criterion (0-1). Higher for physical PPAs and same-market
            instruments.
        temporal_match_score: Score for the temporal match criterion
            (0-1). Higher when vintage matches reporting year.
        geographic_match_score: Score for the geographic match criterion
            (0-1). Higher when instrument region matches consumption
            region.
        no_double_count_score: Score for the no-double-counting
            criterion (0-1). Higher for instruments tracked in
            recognised registries with retirement.
        recognized_registry_score: Score for the recognised registry
            criterion (0-1). Higher for instruments in tier-1
            registries (Green-e, AIB, Ofgem).
        represents_generation_score: Score for the represents-generation
            criterion (0-1). Higher for instruments directly linked
            to verified generation data.
        overall_score: Weighted average of all criterion scores using
            INSTRUMENT_QUALITY_WEIGHTS. Range 0-1.
        passes_threshold: Whether the overall score meets or exceeds
            the minimum quality threshold.
        threshold_used: The quality threshold value used for the
            pass/fail determination.
        assessed_at: UTC timestamp of the assessment.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    instrument_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the assessed instrument",
    )
    unique_claim_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Score for unique claim criterion (0-1)",
    )
    associated_delivery_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Score for associated delivery criterion (0-1)",
    )
    temporal_match_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Score for temporal match criterion (0-1)",
    )
    geographic_match_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Score for geographic match criterion (0-1)",
    )
    no_double_count_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Score for no-double-counting criterion (0-1)",
    )
    recognized_registry_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Score for recognised registry criterion (0-1)",
    )
    represents_generation_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Score for represents-generation criterion (0-1)",
    )
    overall_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Weighted average overall quality score (0-1)",
    )
    passes_threshold: bool = Field(
        ...,
        description="Whether overall score meets minimum threshold",
    )
    threshold_used: Decimal = Field(
        default=DEFAULT_QUALITY_THRESHOLD,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Quality threshold used for pass/fail determination",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the assessment",
    )


class SupplierEmissionFactor(BaseModel):
    """Supplier-specific emission factor for market-based calculations.

    Represents an emission factor disclosed by an energy supplier
    (utility or retailer) based on their generation mix or
    procurement portfolio. Used when supplier-specific data is
    available but no contractual instruments (EACs) are allocated.

    Attributes:
        supplier_id: Unique identifier for the energy supplier.
        name: Human-readable supplier name.
        country: ISO 3166-1 alpha-2 country code where the supplier
            operates.
        ef_kgco2e_per_kwh: Supplier-disclosed emission factor in
            kgCO2e per kWh.
        fuel_mix: Dictionary of energy source proportions in the
            supplier's generation mix (e.g. {"solar": 0.30,
            "natural_gas_ccgt": 0.50, "coal": 0.20}). Values
            should sum to 1.0.
        year: Reporting year the emission factor applies to.
        verified: Whether the emission factor has been independently
            verified or audited by a third party.
        verification_body: Name of the verification body, if verified.
        data_quality_tier: Data quality tier of the supplier factor.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    supplier_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the energy supplier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable supplier name",
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    ef_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Supplier emission factor in kgCO2e per kWh",
    )
    fuel_mix: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Energy source proportions in supplier's generation mix",
    )
    year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year for the emission factor",
    )
    verified: bool = Field(
        default=False,
        description="Whether independently verified by a third party",
    )
    verification_body: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Name of the verification body, if verified",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_2,
        description="Data quality tier of the supplier factor",
    )

    @field_validator("country")
    @classmethod
    def _uppercase_country(cls, v: str) -> str:
        """Normalise country code to uppercase."""
        return v.upper()


class ResidualMixFactor(BaseModel):
    """Residual mix emission factor for a specific region and year.

    Represents the emission intensity of electricity generation
    not claimed by any entity through contractual instruments in
    a given region. Applied to uncovered consumption in market-based
    Scope 2 calculations.

    Attributes:
        region: Geographic region identifier (e.g. US-CAMX, EU-DE,
            APAC-AU, GLOBAL).
        factor_kgco2e_per_kwh: Residual mix emission factor in
            kgCO2e per kWh.
        source: Authority that published the residual mix factor.
        year: Reporting year the factor applies to.
        country_code: ISO 3166-1 alpha-2 country code for the region.
            May be None for multi-country regions or global fallback.
        notes: Optional notes about the factor source, methodology,
            or applicability limitations.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    region: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Geographic region identifier",
    )
    factor_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Residual mix emission factor in kgCO2e per kWh",
    )
    source: ResidualMixSource = Field(
        ...,
        description="Authority that published the residual mix factor",
    )
    year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year the factor applies to",
    )
    country_code: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes about the factor",
    )

    @field_validator("country_code")
    @classmethod
    def _uppercase_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Normalise country code to uppercase."""
        if v is not None:
            return v.upper()
        return v


class EnergyPurchase(BaseModel):
    """Energy purchase record for market-based Scope 2 calculations.

    Represents a single energy purchase for a facility during a
    reporting period, including the consumption quantity, supplier
    information, and any contractual instruments allocated to the
    purchase.

    Attributes:
        purchase_id: Unique identifier for this purchase record.
        facility_id: Reference to the consuming facility.
        energy_type: Type of purchased energy.
        quantity: Consumption quantity in the specified unit.
        unit: Unit of measurement for the quantity.
        region: Geographic region for residual mix factor lookup
            (e.g. US-CAMX, EU-DE, APAC-AU).
        supplier_id: Optional reference to the energy supplier for
            supplier-specific emission factor lookup.
        instruments: List of contractual instruments allocated to
            this purchase. Empty list means uncovered consumption.
        period_start: Start date/time of the purchase period.
        period_end: End date/time of the purchase period.
        data_source: Origin of the consumption data.
        meter_id: Optional utility meter identifier for traceability.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    purchase_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique purchase record identifier",
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
        gt=Decimal("0"),
        description="Consumption quantity in the specified unit",
    )
    unit: EnergyUnit = Field(
        ...,
        description="Unit of measurement for the quantity",
    )
    region: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Geographic region for residual mix factor lookup",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the energy supplier",
    )
    instruments: List[ContractualInstrument] = Field(
        default_factory=list,
        description="Contractual instruments allocated to this purchase",
    )
    period_start: datetime = Field(
        ...,
        description="Start date/time of the purchase period",
    )
    period_end: datetime = Field(
        ...,
        description="End date/time of the purchase period",
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

    @field_validator("instruments")
    @classmethod
    def _validate_instruments_size(
        cls, v: List[ContractualInstrument],
    ) -> List[ContractualInstrument]:
        """Validate that instrument count does not exceed maximum."""
        if len(v) > MAX_INSTRUMENTS_PER_PURCHASE:
            raise ValueError(
                f"Instrument count {len(v)} exceeds maximum "
                f"{MAX_INSTRUMENTS_PER_PURCHASE}"
            )
        return v


class FacilityInfo(BaseModel):
    """Metadata record for a reporting facility in the Scope 2 inventory.

    Represents a single physical facility (building, campus, or site)
    for which market-based Scope 2 emissions are calculated. Each
    facility is mapped to a grid region for residual mix factor
    lookup.

    Attributes:
        facility_id: Unique system identifier for the facility (UUID).
        name: Human-readable facility name or label.
        facility_type: Classification of facility by primary function.
        country_code: ISO 3166-1 alpha-2 country code for the facility.
        grid_region: Grid region identifier for residual mix factor
            lookup (e.g. US-CAMX, EU-DE, APAC-AU).
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
    grid_region: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Grid region identifier for residual mix lookup",
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


class AllocationResult(BaseModel):
    """Result of allocating contractual instruments to an energy purchase.

    Summarises how instruments were allocated to a specific energy
    purchase, including the covered and uncovered portions and
    the per-instrument allocation breakdown.

    Attributes:
        purchase_id: Reference to the energy purchase.
        total_mwh: Total consumption quantity in MWh.
        covered_mwh: Consumption covered by allocated instruments
            in MWh.
        uncovered_mwh: Consumption not covered by instruments in MWh.
        coverage_pct: Percentage of consumption covered (0-100).
        coverage_status: Coverage classification.
        allocations: Per-instrument allocation details, each a
            dictionary with instrument_id, allocated_mwh, and
            emission factor applied.
        allocation_method: Method used for the allocation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    purchase_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the energy purchase",
    )
    total_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total consumption quantity in MWh",
    )
    covered_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Consumption covered by instruments in MWh",
    )
    uncovered_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Consumption not covered by instruments in MWh",
    )
    coverage_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of consumption covered (0-100)",
    )
    coverage_status: CoverageStatus = Field(
        ...,
        description="Coverage classification",
    )
    allocations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-instrument allocation details",
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.PRIORITY_BASED,
        description="Method used for the allocation",
    )


class CoveredEmissionResult(BaseModel):
    """Emission result for consumption covered by a contractual instrument.

    Represents the emissions calculated for a portion of energy
    consumption that is covered by a specific contractual instrument,
    using the instrument's emission factor.

    Attributes:
        instrument_id: Reference to the contractual instrument.
        instrument_type: Type of the contractual instrument.
        mwh_covered: Quantity of consumption covered by this
            instrument in MWh.
        ef_kgco2e_per_kwh: Emission factor applied from the
            instrument in kgCO2e per kWh.
        emissions_kg: Total emissions in kilograms (CO2 only).
        co2e_kg: Total CO2-equivalent emissions in kilograms
            (including CH4 and N2O if applicable).
        energy_source: Energy source of the instrument's generation.
        quality_score: Quality assessment score of the instrument
            (0-1).
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    instrument_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the contractual instrument",
    )
    instrument_type: InstrumentType = Field(
        ...,
        description="Type of the contractual instrument",
    )
    mwh_covered: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Consumption covered by this instrument in MWh",
    )
    ef_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Instrument emission factor in kgCO2e per kWh",
    )
    emissions_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions in kg (CO2 only)",
    )
    co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2-equivalent emissions in kg",
    )
    energy_source: EnergySource = Field(
        ...,
        description="Energy source of the instrument's generation",
    )
    quality_score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Quality assessment score (0-1)",
    )


class UncoveredEmissionResult(BaseModel):
    """Emission result for consumption not covered by instruments.

    Represents the emissions calculated for the portion of energy
    consumption that is not covered by any contractual instrument,
    using the residual mix emission factor for the grid region.

    Attributes:
        mwh_uncovered: Quantity of uncovered consumption in MWh.
        region: Geographic region used for residual mix factor lookup.
        residual_mix_ef_kgco2e_per_kwh: Residual mix emission factor
            applied in kgCO2e per kWh.
        emissions_kg: Total emissions in kilograms (CO2 only).
        co2e_kg: Total CO2-equivalent emissions in kilograms.
        residual_mix_source: Source of the residual mix factor.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    mwh_uncovered: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Quantity of uncovered consumption in MWh",
    )
    region: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Geographic region for residual mix lookup",
    )
    residual_mix_ef_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Residual mix emission factor in kgCO2e per kWh",
    )
    emissions_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions in kg (CO2 only)",
    )
    co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2-equivalent emissions in kg",
    )
    residual_mix_source: ResidualMixSource = Field(
        default=ResidualMixSource.ESTIMATED,
        description="Source of the residual mix factor",
    )


class GasEmissionDetail(BaseModel):
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


class MarketBasedResult(BaseModel):
    """Result of a Scope 2 market-based emission calculation for a facility.

    Contains the complete calculation output including covered and
    uncovered emission breakdowns, instrument allocation details,
    gas-by-gas breakdown, and SHA-256 provenance hash.

    Attributes:
        calculation_id: Unique identifier linking to the request.
        facility_id: Reference to the facility.
        total_mwh: Total energy consumption in MWh.
        covered_mwh: Consumption covered by instruments in MWh.
        uncovered_mwh: Consumption not covered by instruments in MWh.
        coverage_pct: Percentage of consumption covered (0-100).
        covered_emissions_tco2e: Total CO2e from covered consumption
            in tonnes.
        uncovered_emissions_tco2e: Total CO2e from uncovered
            consumption in tonnes.
        total_emissions_tco2e: Total market-based CO2e in tonnes
            (covered + uncovered).
        covered_details: List of per-instrument covered emission
            results.
        uncovered_detail: Uncovered emission result (None if fully
            covered).
        gas_breakdown: List of per-gas emission details.
        calculation_method: Method used for the calculation.
        data_quality_tier: Overall data quality tier.
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
    total_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total energy consumption in MWh",
    )
    covered_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Consumption covered by instruments in MWh",
    )
    uncovered_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Consumption not covered by instruments in MWh",
    )
    coverage_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of consumption covered (0-100)",
    )
    covered_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2e from covered consumption in tonnes",
    )
    uncovered_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2e from uncovered consumption in tonnes",
    )
    total_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total market-based CO2e in tonnes",
    )
    covered_details: List[CoveredEmissionResult] = Field(
        default_factory=list,
        description="Per-instrument covered emission results",
    )
    uncovered_detail: Optional[UncoveredEmissionResult] = Field(
        default=None,
        description="Uncovered emission result (None if fully covered)",
    )
    gas_breakdown: List[GasEmissionDetail] = Field(
        default_factory=list,
        description="Per-gas emission breakdown",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.HYBRID,
        description="Method used for the calculation",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Overall data quality tier",
    )
    provenance_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the calculation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional metadata",
    )


class DualReportingResult(BaseModel):
    """Comparison of location-based and market-based Scope 2 results.

    GHG Protocol Scope 2 Guidance requires dual reporting of both
    location-based and market-based totals. This model captures the
    comparison between the two methods and quantifies the impact of
    renewable energy procurement.

    Attributes:
        dual_id: Unique identifier for this dual reporting record.
        facility_id: Reference to the facility.
        location_based_tco2e: Location-based total emissions in
            tonnes CO2e (from AGENT-MRV-009).
        market_based_tco2e: Market-based total emissions in
            tonnes CO2e (from this agent).
        difference_tco2e: Absolute difference between location and
            market-based totals in tonnes CO2e. Positive means
            market-based is higher; negative means market-based
            is lower (RE procurement reduces emissions).
        difference_pct: Percentage difference relative to the
            location-based total. Negative indicates emission
            reduction from RE procurement.
        re_procurement_impact_tco2e: Emission reduction attributable
            to renewable energy procurement in tonnes CO2e.
            Calculated as the difference between what emissions
            would have been under residual mix and actual
            instrument-based emissions.
        reporting_status: Completeness status of dual reporting.
        reporting_period: Time period for the dual report.
        calculated_at: UTC timestamp of the comparison.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    dual_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique dual reporting record identifier",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the facility",
    )
    location_based_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Location-based total emissions in tCO2e",
    )
    market_based_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Market-based total emissions in tCO2e",
    )
    difference_tco2e: Decimal = Field(
        ...,
        description="Difference (market - location) in tCO2e",
    )
    difference_pct: Decimal = Field(
        ...,
        description="Percentage difference vs location-based",
    )
    re_procurement_impact_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission reduction from RE procurement in tCO2e",
    )
    reporting_status: DualReportingStatus = Field(
        default=DualReportingStatus.COMPLETE,
        description="Completeness status of dual reporting",
    )
    reporting_period: str = Field(
        default="",
        max_length=50,
        description="Time period description (e.g. 2025, 2025-Q1)",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the comparison",
    )


class CalculationRequest(BaseModel):
    """Complete request for a Scope 2 market-based emission calculation.

    Aggregates one or more energy purchase records for a single
    facility, along with shared calculation parameters including
    the GWP source and optional compliance framework checks.

    Attributes:
        calculation_id: Unique identifier for this calculation request.
        tenant_id: Owning tenant identifier for multi-tenancy.
        facility_id: Reference to the target facility.
        energy_purchases: List of energy purchase records, each
            potentially with allocated contractual instruments.
        gwp_source: IPCC Assessment Report for GWP values.
        allocation_method: Method for allocating instruments to
            purchases (if not pre-allocated).
        calculation_method: Calculation methodology to apply.
        compliance_frameworks: Optional list of regulatory framework
            identifiers (e.g. 'GHG_PROTOCOL', 'CSRD', 'CDP',
            'RE100') to run compliance checks against.
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
    energy_purchases: List[EnergyPurchase] = Field(
        default_factory=list,
        description="List of energy purchase records",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.PRIORITY_BASED,
        description="Method for allocating instruments to purchases",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.HYBRID,
        description="Calculation methodology to apply",
    )
    compliance_frameworks: Optional[List[str]] = Field(
        default=None,
        description="Regulatory frameworks for compliance checks",
    )

    @field_validator("energy_purchases")
    @classmethod
    def _validate_purchases_size(
        cls, v: List[EnergyPurchase],
    ) -> List[EnergyPurchase]:
        """Validate that purchase count does not exceed maximum."""
        if len(v) > MAX_ENERGY_PURCHASES_PER_CALC:
            raise ValueError(
                f"Energy purchase count {len(v)} exceeds maximum "
                f"{MAX_ENERGY_PURCHASES_PER_CALC}"
            )
        return v


class BatchCalculationRequest(BaseModel):
    """Batch request for multiple Scope 2 market-based calculations.

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


class BatchCalculationResult(BaseModel):
    """Result of a batch Scope 2 market-based calculation.

    Aggregates results from all individual calculations in a batch
    with portfolio-level totals and coverage metrics.

    Attributes:
        batch_id: Unique identifier linking to the batch request.
        results: List of individual market-based calculation results.
        total_co2e_tonnes: Portfolio-level total CO2e in tonnes.
        facility_count: Number of unique facilities in the batch.
        average_coverage_pct: Average instrument coverage across
            all facilities (0-100).
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
    results: List[MarketBasedResult] = Field(
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
    average_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Average instrument coverage across facilities (0-100)",
    )
    provenance_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 provenance hash for batch audit trail",
    )


class ComplianceCheckResult(BaseModel):
    """Result of a regulatory compliance check for a calculation.

    Evaluates a completed market-based calculation against a specific
    regulatory framework and reports findings and recommendations.

    Attributes:
        check_id: Unique identifier for this compliance check.
        calculation_id: Reference to the calculation being checked.
        framework: Regulatory framework identifier (e.g.
            'GHG_PROTOCOL', 'CSRD', 'CDP', 'RE100', 'ISO_14064').
        status: Overall compliance status.
        findings: List of specific compliance findings, each
            describing a requirement and its assessment.
        recommendations: List of recommended actions to achieve
            or maintain compliance.
        coverage_requirement: Minimum instrument coverage required
            by the framework (0-100), if applicable.
        actual_coverage: Actual instrument coverage achieved (0-100).
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
    coverage_requirement: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Minimum coverage required by framework (0-100)",
    )
    actual_coverage: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Actual instrument coverage achieved (0-100)",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the compliance check",
    )


class InstrumentValidationResult(BaseModel):
    """Result of validating a contractual instrument for market-based use.

    Combines the quality assessment with a validity determination
    and lists any issues found during validation.

    Attributes:
        instrument_id: Reference to the validated instrument.
        quality_assessment: Full quality assessment of the instrument.
        is_valid: Whether the instrument passes all validation checks
            (quality threshold, vintage validity, status, registry).
        issues: List of validation issues found. Empty list means
            no issues and the instrument is valid.
        vintage_valid: Whether the vintage year is within the
            allowable window for the instrument type.
        status_valid: Whether the instrument status allows allocation
            (must be active or retired for the reporting period).
        validated_at: UTC timestamp of the validation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    instrument_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the validated instrument",
    )
    quality_assessment: InstrumentQualityAssessment = Field(
        ...,
        description="Full quality assessment of the instrument",
    )
    is_valid: bool = Field(
        ...,
        description="Whether the instrument passes all validation checks",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Validation issues found (empty if valid)",
    )
    vintage_valid: bool = Field(
        default=True,
        description="Whether vintage year is within allowable window",
    )
    status_valid: bool = Field(
        default=True,
        description="Whether instrument status allows allocation",
    )
    validated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the validation",
    )


class UncertaintyRequest(BaseModel):
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


class UncertaintyResult(BaseModel):
    """Result of uncertainty quantification for a market-based calculation.

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


class AggregationResult(BaseModel):
    """Aggregated emission result across multiple calculations.

    Provides portfolio-level or group-level totals for Scope 2
    market-based emissions, grouped by a specified dimension
    (facility, instrument type, energy source, or time period).

    Attributes:
        aggregation_id: Unique identifier for this aggregation.
        group_by: Dimension used for grouping (e.g. 'facility',
            'instrument_type', 'energy_source', 'month').
        period: Reporting period description (e.g. '2025',
            '2025-Q1', '2025-01').
        total_co2e_tonnes: Aggregated total CO2e in tonnes.
        total_covered_mwh: Aggregated covered consumption in MWh.
        total_uncovered_mwh: Aggregated uncovered consumption in MWh.
        average_coverage_pct: Average coverage across the group
            (0-100).
        facility_count: Number of unique facilities in the group.
        instrument_count: Number of unique instruments in the group.
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
    total_covered_mwh: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregated covered consumption in MWh",
    )
    total_uncovered_mwh: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregated uncovered consumption in MWh",
    )
    average_coverage_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Average coverage across the group (0-100)",
    )
    facility_count: int = Field(
        ...,
        ge=0,
        description="Number of unique facilities in the group",
    )
    instrument_count: int = Field(
        default=0,
        ge=0,
        description="Number of unique instruments in the group",
    )
    details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-group detail records",
    )
