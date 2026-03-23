# -*- coding: utf-8 -*-
"""
Fugitive Emissions Agent Data Models - AGENT-MRV-005

Pydantic v2 data models for the Fugitive Emissions Agent SDK covering
GHG Protocol Scope 1 fugitive emission calculations including:
- Equipment leak detection and repair (LDAR) per EPA Method 21 / OGI
- Oil & gas production, processing, transmission, and distribution sources
- Coal mine methane (underground, surface, post-mining) per IPCC Vol 2 Ch 4
- Wastewater treatment emissions (industrial and municipal)
- Pneumatic device emissions (high-bleed, low-bleed, intermittent)
- Tank storage losses per AP-42 Chapter 7 (fixed roof, floating roof)
- 5 calculation methods (average EF, screening ranges, correlation
  equation, engineering estimate, direct measurement)
- EPA component-level average emission factors for 9 component types
  across 4 service types
- EPA correlation equation coefficients for screening-to-mass conversion
- IPCC coal mine methane factors by mining type and coal rank
- IPCC wastewater methane correction factors by treatment type
- Monte Carlo uncertainty quantification
- Multi-framework regulatory compliance (GHG Protocol, ISO 14064,
  CSRD, EPA 40 CFR Part 98, EU ETS, OGMP 2.0)
- SHA-256 provenance chain for complete audit trails

Enumerations (16):
    - FugitiveSourceCategory, FugitiveSourceType, ComponentType,
      ServiceType, EmissionGas, CalculationMethod, EmissionFactorSource,
      GWPSource, SurveyType, LeakStatus, CoalRank, WastewaterType,
      ComplianceStatus, ReportingPeriod, UnitType, TankType

Constants:
    - GWP_VALUES: IPCC AR4/AR5/AR6/AR6-20yr GWP values (Decimal)
    - EPA_COMPONENT_EMISSION_FACTORS: EPA average EF by (component, service)
    - EPA_CORRELATION_COEFFICIENTS: screening-to-mass correlation coefficients
    - IPCC_COAL_EMISSION_FACTORS: in-situ CH4 by mining type and coal rank
    - WASTEWATER_MCF: Methane correction factors by treatment type
    - PNEUMATIC_RATES_M3_PER_DAY: Whole-gas vent rates for pneumatic devices
    - SOURCE_CATEGORY_MAP: Mapping from source category to source types
    - SOURCE_DEFAULT_GASES: Default gas profiles per source type

Data Models (16):
    - FugitiveSourceInfo, ComponentRecord, EmissionFactorRecord,
      SurveyRecord, LeakRecord, RepairRecord, CalculationRequest,
      CalculationResult, CalculationDetailResult, BatchCalculationRequest,
      BatchCalculationResult, UncertaintyRequest, UncertaintyResult,
      ComplianceCheckResult, AggregationRequest, AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
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

#: Maximum number of components in a single calculation request.
MAX_COMPONENTS_PER_CALC: int = 50_000

#: Maximum number of survey records per facility.
MAX_SURVEYS_PER_FACILITY: int = 5_000

#: Maximum number of leak records per survey.
MAX_LEAKS_PER_SURVEY: int = 10_000

#: Default LDAR leak threshold in parts per million (EPA Method 21).
DEFAULT_LEAK_THRESHOLD_PPM: int = 10_000

#: Default repair deadline in calendar days from detection date.
DEFAULT_REPAIR_DEADLINE_DAYS: int = 15

#: Maximum delay of repair extension in calendar days.
MAX_DELAY_OF_REPAIR_DAYS: int = 365


# =============================================================================
# Enumerations (16)
# =============================================================================


class FugitiveSourceCategory(str, Enum):
    """Broad classification of fugitive emission sources by sector and activity.

    Groups fugitive emission source types for reporting aggregation and
    to determine applicable default emission factors, survey requirements,
    and regulatory reporting obligations.

    OIL_GAS_PRODUCTION: Upstream oil and gas extraction and wellsite
        processing, including wellheads, separators, and dehydrators.
    OIL_GAS_PROCESSING: Midstream gas processing plant operations
        including acid gas removal and glycol dehydration.
    GAS_TRANSMISSION: High-pressure natural gas pipeline transmission
        including compressor stations and metering facilities.
    GAS_DISTRIBUTION: Low-pressure natural gas distribution networks
        including service lines, meters, and regulators.
    CRUDE_OIL: Crude oil transportation and terminal operations.
    LNG: Liquefied natural gas facility operations.
    COAL_UNDERGROUND: Underground coal mining methane emissions.
    COAL_SURFACE: Surface (open-cut) coal mining methane emissions.
    COAL_POST_MINING: Post-mining methane emissions from abandoned
        mines and coal handling/processing.
    WASTEWATER_INDUSTRIAL: Industrial wastewater treatment facilities.
    WASTEWATER_MUNICIPAL: Municipal wastewater treatment facilities.
    EQUIPMENT_LEAKS: Component-level fugitive leaks from process
        equipment (valves, pumps, compressors, connectors, flanges).
    TANK_STORAGE: Hydrocarbon storage tank losses (breathing and
        working losses) per AP-42 Chapter 7.
    PNEUMATIC_DEVICES: Natural gas-actuated pneumatic controllers
        and pumps venting whole gas.
    OTHER: Fugitive emission sources not classified elsewhere.
    """

    OIL_GAS_PRODUCTION = "oil_gas_production"
    OIL_GAS_PROCESSING = "oil_gas_processing"
    GAS_TRANSMISSION = "gas_transmission"
    GAS_DISTRIBUTION = "gas_distribution"
    CRUDE_OIL = "crude_oil"
    LNG = "lng"
    COAL_UNDERGROUND = "coal_underground"
    COAL_SURFACE = "coal_surface"
    COAL_POST_MINING = "coal_post_mining"
    WASTEWATER_INDUSTRIAL = "wastewater_industrial"
    WASTEWATER_MUNICIPAL = "wastewater_municipal"
    EQUIPMENT_LEAKS = "equipment_leaks"
    TANK_STORAGE = "tank_storage"
    PNEUMATIC_DEVICES = "pneumatic_devices"
    OTHER = "other"


class FugitiveSourceType(str, Enum):
    """Specific fugitive emission source type identifiers (25 values).

    Covers all major fugitive emission sources encountered in Scope 1
    GHG inventories across oil and gas operations, coal mining,
    wastewater treatment, and industrial equipment leaks. Each source
    type has associated default emission factors, applicable gases,
    and regulatory references.

    Naming follows EPA Subpart W (40 CFR Part 98), IPCC 2006 Guidelines
    Volume 2 Chapter 4, and API Compendium conventions for
    cross-framework compatibility.
    """

    # Oil & gas production (upstream)
    WELLHEAD = "wellhead"
    SEPARATOR = "separator"
    DEHYDRATOR = "dehydrator"

    # Pneumatic controllers by bleed rate
    PNEUMATIC_CONTROLLER_HIGH = "pneumatic_controller_high"
    PNEUMATIC_CONTROLLER_LOW = "pneumatic_controller_low"
    PNEUMATIC_CONTROLLER_INTERMITTENT = "pneumatic_controller_intermittent"

    # Compressors
    COMPRESSOR_CENTRIFUGAL = "compressor_centrifugal"
    COMPRESSOR_RECIPROCATING = "compressor_reciprocating"

    # Gas processing
    ACID_GAS_REMOVAL = "acid_gas_removal"
    GLYCOL_DEHYDRATOR = "glycol_dehydrator"

    # Gas transmission and distribution
    PIPELINE_MAIN = "pipeline_main"
    PIPELINE_SERVICE = "pipeline_service"
    METER_REGULATOR = "meter_regulator"

    # Tank storage
    TANK_FIXED_ROOF = "tank_fixed_roof"
    TANK_FLOATING_ROOF = "tank_floating_roof"

    # Coal mining
    COAL_MINE_UNDERGROUND = "coal_mine_underground"
    COAL_MINE_SURFACE = "coal_mine_surface"
    COAL_HANDLING = "coal_handling"
    ABANDONED_MINE = "abandoned_mine"

    # Wastewater treatment
    WASTEWATER_LAGOON = "wastewater_lagoon"
    WASTEWATER_DIGESTER = "wastewater_digester"
    WASTEWATER_AEROBIC = "wastewater_aerobic"

    # Equipment leak components
    VALVE_GAS = "valve_gas"
    PUMP_SEAL = "pump_seal"
    COMPRESSOR_SEAL = "compressor_seal"
    FLANGE_CONNECTOR = "flange_connector"


class ComponentType(str, Enum):
    """Classification of process equipment components for LDAR monitoring.

    Component type determines the applicable EPA average emission factor,
    screening value correlation equation, and survey frequency requirements.
    Each component type has characteristic leak rates and emission profiles
    that vary by service type (gas, light liquid, heavy liquid).

    Naming follows EPA Protocol for Equipment Leak Emission Estimates
    (EPA-453/R-95-017) and 40 CFR Part 60 Subpart VVa conventions.

    VALVE: Manual or automated valves in process piping.
    PUMP: Rotating shaft seals on centrifugal and positive displacement pumps.
    COMPRESSOR: Shaft seals on gas compressors.
    PRESSURE_RELIEF_DEVICE: Spring-loaded or rupture-disc relief valves.
    CONNECTOR: Flanged, threaded, or other pipe connections.
    OPEN_ENDED_LINE: Piping open to atmosphere (e.g. drains, vents, purges).
    SAMPLING_CONNECTION: Ports for process stream sampling.
    FLANGE: Bolted flange connections (tracked separately from connectors
        in some LDAR programs).
    OTHER: Components not classified in the standard EPA categories.
    """

    VALVE = "valve"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    PRESSURE_RELIEF_DEVICE = "pressure_relief_device"
    CONNECTOR = "connector"
    OPEN_ENDED_LINE = "open_ended_line"
    SAMPLING_CONNECTION = "sampling_connection"
    FLANGE = "flange"
    OTHER = "other"


class ServiceType(str, Enum):
    """Process stream service classification for component emission factors.

    The service type of a component determines the applicable EPA average
    emission factor from EPA-453/R-95-017. Emission factors vary
    significantly by service type due to differences in fluid properties
    (volatility, viscosity, density) and leak behaviour.

    GAS: Components in gaseous service (natural gas, process gas,
        fuel gas, vent gas, compressed air).
    LIGHT_LIQUID: Components in light liquid service (condensate,
        light crude, gasoline, naphtha, kerosene, light fuel oils).
        Defined as vapour pressure > 0.3 kPa at operating conditions.
    HEAVY_LIQUID: Components in heavy liquid service (heavy crude,
        heavy fuel oil, asphalt, lube oil, glycol, amine solutions).
        Defined as vapour pressure <= 0.3 kPa at operating conditions.
    HYDROGEN: Components in hydrogen service. Hydrogen has distinct
        leak characteristics due to its low molecular weight and
        high diffusivity.
    """

    GAS = "gas"
    LIGHT_LIQUID = "light_liquid"
    HEAVY_LIQUID = "heavy_liquid"
    HYDROGEN = "hydrogen"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in fugitive emission calculations.

    CH4: Methane - primary fugitive emission from oil and gas operations,
        coal mining, and wastewater treatment. Major contributor to GHG
        inventory due to high GWP.
    CO2: Carbon dioxide - released from acid gas removal, some equipment
        leaks with high CO2 gas composition, and wastewater treatment.
    N2O: Nitrous oxide - emitted from wastewater treatment processes,
        particularly nitrification and denitrification.
    VOC: Volatile Organic Compounds - emitted from equipment leaks and
        tank storage losses. Tracked for air quality compliance but
        not included in CO2e totals unless specific GWP is assigned.
    """

    CH4 = "CH4"
    CO2 = "CO2"
    N2O = "N2O"
    VOC = "VOC"


class CalculationMethod(str, Enum):
    """Methodology for calculating fugitive emissions from equipment leaks.

    AVERAGE_EMISSION_FACTOR: Applies EPA average emission factors
        (kg/hr/component) to component counts by type and service.
        Simplest Tier 1 approach. EPA-453/R-95-017 Table 2-1.
    SCREENING_RANGES: Uses screening value ranges from portable
        analyzers to assign emission rates from screening range tables.
        More accurate than average EF when screening data is available.
    CORRELATION_EQUATION: Applies EPA correlation equations that relate
        measured screening value (ppm) to mass emission rate (kg/hr).
        Highest accuracy for individual component leak quantification.
        EPA-453/R-95-017 Table 2-8.
    ENGINEERING_ESTIMATE: Uses engineering calculations for specific
        equipment types (e.g. pneumatic devices, compressor rod packing,
        tank breathing/working losses per AP-42). Not based on screening
        data.
    DIRECT_MEASUREMENT: Uses Hi-Flow sampler, bagging, or other direct
        measurement techniques to quantify individual leak rates.
        Highest accuracy when measurement data is available.
    """

    AVERAGE_EMISSION_FACTOR = "AVERAGE_EMISSION_FACTOR"
    SCREENING_RANGES = "SCREENING_RANGES"
    CORRELATION_EQUATION = "CORRELATION_EQUATION"
    ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"


class EmissionFactorSource(str, Enum):
    """Source authority for fugitive emission factor values.

    EPA: US Environmental Protection Agency (40 CFR Part 98 Subpart W,
        EPA-453/R-95-017, AP-42 Chapter 7). Primary source for
        equipment leak and tank loss factors.
    IPCC: IPCC 2006 Guidelines for National GHG Inventories, Volume 2
        Chapter 4 (Fugitive Emissions from Oil and Natural Gas) and
        Volume 5 Chapter 5 (Waste - Wastewater Treatment).
    DEFRA: UK Department for Environment, Food and Rural Affairs
        conversion factors for company reporting of fugitive emissions.
    EU_ETS: European Union Emissions Trading System Monitoring and
        Reporting Regulation (EU MRR) fugitive emission factors.
    API: American Petroleum Institute Compendium of Greenhouse Gas
        Emissions Methodologies for the Oil and Natural Gas Industry.
    CUSTOM: Organization-specific or facility-measured emission factors.
        Requires documentation and third-party verification for
        regulatory compliance.
    """

    EPA = "EPA"
    IPCC = "IPCC"
    DEFRA = "DEFRA"
    EU_ETS = "EU_ETS"
    API = "API"
    CUSTOM = "CUSTOM"


class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for GWP conversion factors.

    AR4: Fourth Assessment Report (2007). GWP-100yr. CH4=25, N2O=298.
    AR5: Fifth Assessment Report (2014). GWP-100yr. CH4=28, N2O=265.
    AR6: Sixth Assessment Report (2021). GWP-100yr. CH4=29.8, N2O=273.
        Includes climate-carbon feedback.
    AR6_20YR: Sixth Assessment Report (2021). GWP-20yr timeframe.
        CH4=82.5, N2O=273. Highlights short-lived climate pollutant
        impact over a 20-year horizon.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class SurveyType(str, Enum):
    """Leak detection survey methodology type for LDAR programs.

    OGI: Optical Gas Imaging using infrared cameras (e.g. FLIR GF-series).
        Qualitative detection method per EPA OOOOa. Can scan large
        areas rapidly but does not directly quantify leak rate.
    METHOD_21: EPA Reference Method 21 using portable analyzers (TVA,
        OVA, phx21). Quantitative screening values in ppm. Required for
        regulatory LDAR at SOCMI and petroleum refinery facilities.
    AVO: Audio/Visual/Olfactory inspection. Lowest-cost method for
        detecting large leaks. Does not provide quantitative data.
    HI_FLOW: Hi-Flow Sampler direct measurement of individual leak
        rates in standard cubic feet per hour (scfh) or liters per
        minute (lpm). Highest quantitative accuracy per component.
    """

    OGI = "OGI"
    METHOD_21 = "METHOD_21"
    AVO = "AVO"
    HI_FLOW = "HI_FLOW"


class LeakStatus(str, Enum):
    """Current leak status of an equipment component.

    NO_LEAK: No leak detected. Screening value below threshold or
        OGI shows no visible emissions.
    LEAK_DETECTED: Leak confirmed by survey. Screening value at or
        above threshold, or OGI visualization confirmed.
    REPAIR_PENDING: Leak detected and repair has been scheduled but
        not yet completed. Component is within the regulatory repair
        deadline window.
    REPAIRED: Leak has been repaired and post-repair verification
        confirms screening value below threshold.
    DELAY_OF_REPAIR: Repair cannot be completed within the standard
        deadline due to technical infeasibility or process constraints.
        Requires documented justification per 40 CFR 60.482/63.163.
    """

    NO_LEAK = "no_leak"
    LEAK_DETECTED = "leak_detected"
    REPAIR_PENDING = "repair_pending"
    REPAIRED = "repaired"
    DELAY_OF_REPAIR = "delay_of_repair"


class CoalRank(str, Enum):
    """Classification of coal by rank for methane content estimation.

    Coal rank determines the in-situ methane content used to estimate
    fugitive methane emissions from mining and post-mining activities.
    Higher-rank coals generally have higher methane content due to
    greater thermal maturity and coalification.

    ANTHRACITE: Highest rank. In-situ CH4 ~18.0 m3/tonne. Low
        volatile matter, high carbon content, hard and dense.
    BITUMINOUS: High rank. In-situ CH4 ~10.0 m3/tonne. Most common
        coal for power generation and coking.
    SUBBITUMINOUS: Medium rank. In-situ CH4 ~3.0 m3/tonne. Lower
        heating value, higher moisture than bituminous.
    LIGNITE: Lowest rank. In-situ CH4 ~1.0 m3/tonne. Highest
        moisture content, lowest heating value.
    """

    ANTHRACITE = "anthracite"
    BITUMINOUS = "bituminous"
    SUBBITUMINOUS = "subbituminous"
    LIGNITE = "lignite"


class WastewaterType(str, Enum):
    """Classification of wastewater treatment system types.

    Treatment type determines the Methane Correction Factor (MCF) used
    to estimate CH4 emissions from wastewater. Anaerobic conditions
    produce significantly higher methane than aerobic treatment.

    AEROBIC: Fully aerobic treatment (activated sludge, trickling
        filter, oxidation ditch). MCF = 0.0 (no significant CH4).
    ANAEROBIC_LAGOON: Uncovered anaerobic lagoon without biogas
        capture. MCF = 0.8. Major CH4 source.
    ANAEROBIC_DIGESTER: Enclosed anaerobic digester with or without
        biogas capture. MCF = 0.8 for uncontrolled venting;
        substantially lower with biogas recovery.
    FACULTATIVE: Facultative lagoon with aerobic surface and
        anaerobic bottom zones. MCF = 0.2.
    SEPTIC: Septic system with anaerobic conditions. MCF = 0.5.
    """

    AEROBIC = "aerobic"
    ANAEROBIC_LAGOON = "anaerobic_lagoon"
    ANAEROBIC_DIGESTER = "anaerobic_digester"
    FACULTATIVE = "facultative"
    SEPTIC = "septic"


class ComplianceStatus(str, Enum):
    """Compliance check result status for a regulatory framework.

    COMPLIANT: All applicable requirements are fully met.
    NON_COMPLIANT: One or more requirements are not met.
    PARTIAL: Some requirements are met, others require attention.
    NOT_CHECKED: Compliance has not been evaluated against this framework.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_CHECKED = "not_checked"


class ReportingPeriod(str, Enum):
    """Temporal granularity for fugitive emission reporting aggregation.

    MONTHLY: Calendar month aggregation.
    QUARTERLY: Calendar quarter (Q1-Q4) aggregation.
    ANNUAL: Full calendar or fiscal year aggregation.
    """

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class UnitType(str, Enum):
    """Physical unit categories for activity data quantities.

    MASS: Weight-based units (kg, tonnes, lbs, short tons).
    VOLUME: Volumetric units (standard cubic meters, standard cubic feet,
        liters, gallons, barrels).
    COUNT: Discrete component counts (number of valves, pumps, devices).
    TIME: Duration units (hours, days, years) for operating time or
        leak duration tracking.
    """

    MASS = "mass"
    VOLUME = "volume"
    COUNT = "count"
    TIME = "time"


class TankType(str, Enum):
    """Classification of hydrocarbon storage tanks for loss estimation.

    Tank type determines the applicable AP-42 Chapter 7 loss model
    and the dominant emission mechanisms (breathing loss vs. working
    loss vs. rim seal loss).

    FIXED_ROOF: Vertical or horizontal fixed-roof tank. Subject to
        breathing losses (thermal expansion/contraction) and working
        losses (filling/emptying). AP-42 Section 7.1 equations.
    FLOATING_ROOF_EXTERNAL: External floating roof tank with roof
        directly exposed to weather. Primary losses from rim seal,
        fitting, and deck seam emissions. Lowest overall loss rate.
    FLOATING_ROOF_INTERNAL: Internal (covered) floating roof tank
        with fixed outer roof and floating inner roof. Combines some
        fixed-roof and floating-roof loss mechanisms.
    PRESSURIZED: Pressurized storage tank (sphere, bullet). Minimal
        breathing/working losses under normal operation. Losses occur
        primarily during pressure relief events.
    """

    FIXED_ROOF = "fixed_roof"
    FLOATING_ROOF_EXTERNAL = "floating_roof_external"
    FLOATING_ROOF_INTERNAL = "floating_roof_internal"
    PRESSURIZED = "pressurized"


# =============================================================================
# GWP Values Lookup Table
# =============================================================================

#: Global Warming Potential values for fugitive emission gases by IPCC AR.
#: Units: kg CO2e per kg gas (dimensionless multiplier).
#: Decimal precision for zero-hallucination regulatory calculations.
#: Sources:
#:   AR4: IPCC Fourth Assessment Report (2007), Table 2.14.
#:   AR5: IPCC Fifth Assessment Report (2014), Table 8.A.1.
#:   AR6: IPCC Sixth Assessment Report (2021), Table 7.15.
#:   AR6_20YR: AR6 GWP-20yr timeframe, Table 7.15.
#:
#: Note: VOC does not have a standard GWP and is excluded from CO2e totals.
GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CH4": Decimal("25"),
        "CO2": Decimal("1"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CH4": Decimal("28"),
        "CO2": Decimal("1"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CH4": Decimal("29.8"),
        "CO2": Decimal("1"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CH4": Decimal("82.5"),
        "CO2": Decimal("1"),
        "N2O": Decimal("273"),
    },
}


# =============================================================================
# EPA Component-Level Average Emission Factors
# =============================================================================

#: EPA average emission factors by (component_type, service_type).
#: Units: kg/hr/component (total organic compounds including methane).
#: Source: EPA-453/R-95-017 "Protocol for Equipment Leak Emission Estimates",
#:         Table 2-1 (Average Emission Factors for EPA Correlation Method).
#: Also referenced in 40 CFR Part 98 Subpart W, Table W-1A.
#:
#: Key: (ComponentType value, ServiceType value) -> Decimal kg/hr.
#: Not all (component, service) combinations have published factors.
#: Only combinations with EPA-published values are included.
#:
#: Factors represent total organic compound (TOC) emission rates.
#: To convert to specific gas (CH4, VOC), multiply by the appropriate
#: weight fraction in the process stream.
EPA_COMPONENT_EMISSION_FACTORS: Dict[
    Tuple[str, str], Decimal
] = {
    # Valves
    ("valve", "gas"): Decimal("0.00597"),
    ("valve", "light_liquid"): Decimal("0.00403"),
    ("valve", "heavy_liquid"): Decimal("0.00023"),
    ("valve", "hydrogen"): Decimal("0.00597"),
    # Pumps
    ("pump", "gas"): Decimal("0.01190"),
    ("pump", "light_liquid"): Decimal("0.01140"),
    ("pump", "heavy_liquid"): Decimal("0.00862"),
    ("pump", "hydrogen"): Decimal("0.01190"),
    # Compressors
    ("compressor", "gas"): Decimal("0.22800"),
    ("compressor", "light_liquid"): Decimal("0.22800"),
    ("compressor", "heavy_liquid"): Decimal("0.22800"),
    ("compressor", "hydrogen"): Decimal("0.22800"),
    # Pressure relief devices
    ("pressure_relief_device", "gas"): Decimal("0.10400"),
    ("pressure_relief_device", "light_liquid"): Decimal("0.10400"),
    ("pressure_relief_device", "heavy_liquid"): Decimal("0.10400"),
    ("pressure_relief_device", "hydrogen"): Decimal("0.10400"),
    # Connectors
    ("connector", "gas"): Decimal("0.00183"),
    ("connector", "light_liquid"): Decimal("0.00183"),
    ("connector", "heavy_liquid"): Decimal("0.00026"),
    ("connector", "hydrogen"): Decimal("0.00183"),
    # Open-ended lines
    ("open_ended_line", "gas"): Decimal("0.00170"),
    ("open_ended_line", "light_liquid"): Decimal("0.00170"),
    ("open_ended_line", "heavy_liquid"): Decimal("0.00170"),
    ("open_ended_line", "hydrogen"): Decimal("0.00170"),
    # Sampling connections
    ("sampling_connection", "gas"): Decimal("0.01500"),
    ("sampling_connection", "light_liquid"): Decimal("0.01500"),
    ("sampling_connection", "heavy_liquid"): Decimal("0.01500"),
    ("sampling_connection", "hydrogen"): Decimal("0.01500"),
    # Flanges
    ("flange", "gas"): Decimal("0.00083"),
    ("flange", "light_liquid"): Decimal("0.00083"),
    ("flange", "heavy_liquid"): Decimal("0.00017"),
    ("flange", "hydrogen"): Decimal("0.00083"),
    # Other components (use connector defaults)
    ("other", "gas"): Decimal("0.00183"),
    ("other", "light_liquid"): Decimal("0.00183"),
    ("other", "heavy_liquid"): Decimal("0.00026"),
    ("other", "hydrogen"): Decimal("0.00183"),
}


# =============================================================================
# EPA Correlation Equation Coefficients
# =============================================================================

#: EPA correlation equation coefficients for screening-to-mass
#: emission rate conversion.
#: Key: (component_type, service_type) -> (slope, intercept) for
#: log10(kg/hr) = slope * log10(ppmv) + intercept.
#: Source: EPA-453/R-95-017, Tables 2-8 through 2-11.
EPA_CORRELATION_COEFFICIENTS: Dict[
    Tuple[str, str], Tuple[Decimal, Decimal]
] = {
    ("valve", "gas"): (Decimal("0.7240"), Decimal("-6.5850")),
    ("valve", "light_liquid"): (Decimal("0.6930"), Decimal("-6.2550")),
    ("pump", "gas"): (Decimal("0.8530"), Decimal("-6.1440")),
    ("pump", "light_liquid"): (Decimal("0.8480"), Decimal("-5.8250")),
    ("compressor", "gas"): (Decimal("0.7060"), Decimal("-5.2310")),
    ("connector", "gas"): (Decimal("0.6520"), Decimal("-6.4240")),
    ("connector", "light_liquid"): (Decimal("0.6320"), Decimal("-6.1520")),
    ("flange", "gas"): (Decimal("0.5870"), Decimal("-6.7150")),
    ("open_ended_line", "gas"): (Decimal("0.6230"), Decimal("-5.9580")),
    ("other", "gas"): (Decimal("0.6520"), Decimal("-6.4240")),
}


# =============================================================================
# Coal Mine Methane Content by Rank and Mining Type
# =============================================================================

#: IPCC default methane emission factors for coal mining by mining type
#: and coal rank.
#: Units: standard cubic meters CH4 per tonne of coal (m3/t).
#: Source: IPCC 2006 Guidelines, Volume 2, Chapter 4, Table 4.1.3.
#:
#: Used as a default when mine-specific measurements are not available.
#: Actual in-situ content varies significantly by seam depth, geology,
#: and geological history. Mine-specific data should be preferred when
#: available (Tier 2/3 approach).
IPCC_COAL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "underground": {
        "anthracite": Decimal("18.0"),
        "bituminous": Decimal("10.0"),
        "subbituminous": Decimal("5.0"),
        "lignite": Decimal("2.0"),
    },
    "surface": {
        "anthracite": Decimal("1.2"),
        "bituminous": Decimal("0.9"),
        "subbituminous": Decimal("0.6"),
        "lignite": Decimal("0.3"),
    },
    "post_mining": {
        "anthracite": Decimal("2.5"),
        "bituminous": Decimal("1.5"),
        "subbituminous": Decimal("0.8"),
        "lignite": Decimal("0.4"),
    },
}

#: Legacy alias for backward compatibility.
COAL_METHANE_FACTORS: Dict[str, Decimal] = {
    "anthracite": Decimal("18.0"),
    "bituminous": Decimal("10.0"),
    "subbituminous": Decimal("3.0"),
    "lignite": Decimal("1.0"),
}


# =============================================================================
# Wastewater Methane Correction Factors
# =============================================================================

#: Methane correction factors (MCF) by wastewater treatment type.
#: Dimensionless fraction (0.0 - 1.0) representing the proportion of
#: maximum methane-producing capacity that is realised under actual
#: treatment conditions.
#: Source: IPCC 2006 Guidelines, Volume 5, Chapter 6, Table 6.3.
#:
#: MCF = 0 for fully aerobic treatment; MCF = 0.8 for fully anaerobic
#: treatment without biogas recovery; intermediate values for
#: facultative and septic systems.
WASTEWATER_MCF: Dict[str, Decimal] = {
    "aerobic": Decimal("0.0"),
    "anaerobic_lagoon": Decimal("0.8"),
    "anaerobic_digester": Decimal("0.8"),
    "facultative": Decimal("0.2"),
    "septic": Decimal("0.5"),
}


# =============================================================================
# Pneumatic Device Emission Rates
# =============================================================================

#: Whole-gas vent rates for natural gas-actuated pneumatic devices.
#: Units: standard cubic meters per day (m3/day) per device.
#: Source: EPA GHG Inventory (Annex 3, Table 3.6-2) and
#:         EPA-453/R-95-017, based on GRI/EPA 1996 study data.
#:
#: HIGH_BLEED: Continuous-bleed controllers with vent rate > 6 scfh
#:     (approximately 37.8 m3/day whole gas).
#: LOW_BLEED: Continuous-bleed controllers with vent rate <= 6 scfh
#:     (approximately 0.945 m3/day whole gas).
#: INTERMITTENT: Intermittent-vent (snap-acting) controllers that vent
#:     during actuation events (approximately 9.18 m3/day average).
PNEUMATIC_RATES_M3_PER_DAY: Dict[str, Decimal] = {
    "HIGH_BLEED": Decimal("37.8"),
    "LOW_BLEED": Decimal("0.945"),
    "INTERMITTENT": Decimal("9.18"),
}


# =============================================================================
# Source Category to Source Type Mapping
# =============================================================================

#: Mapping from FugitiveSourceCategory to the set of applicable
#: FugitiveSourceType values. Used for validation and reporting aggregation.
SOURCE_CATEGORY_MAP: Dict[str, List[str]] = {
    "oil_gas_production": [
        "wellhead",
        "separator",
        "dehydrator",
        "pneumatic_controller_high",
        "pneumatic_controller_low",
        "pneumatic_controller_intermittent",
    ],
    "oil_gas_processing": [
        "acid_gas_removal",
        "glycol_dehydrator",
        "compressor_centrifugal",
        "compressor_reciprocating",
    ],
    "gas_transmission": [
        "pipeline_main",
        "compressor_centrifugal",
        "compressor_reciprocating",
        "meter_regulator",
    ],
    "gas_distribution": [
        "pipeline_main",
        "pipeline_service",
        "meter_regulator",
    ],
    "crude_oil": [
        "tank_fixed_roof",
        "tank_floating_roof",
        "separator",
    ],
    "lng": [
        "compressor_centrifugal",
        "compressor_reciprocating",
        "tank_fixed_roof",
    ],
    "coal_underground": [
        "coal_mine_underground",
    ],
    "coal_surface": [
        "coal_mine_surface",
    ],
    "coal_post_mining": [
        "coal_handling",
        "abandoned_mine",
    ],
    "wastewater_industrial": [
        "wastewater_lagoon",
        "wastewater_digester",
        "wastewater_aerobic",
    ],
    "wastewater_municipal": [
        "wastewater_lagoon",
        "wastewater_digester",
        "wastewater_aerobic",
    ],
    "equipment_leaks": [
        "valve_gas",
        "pump_seal",
        "compressor_seal",
        "flange_connector",
    ],
    "tank_storage": [
        "tank_fixed_roof",
        "tank_floating_roof",
    ],
    "pneumatic_devices": [
        "pneumatic_controller_high",
        "pneumatic_controller_low",
        "pneumatic_controller_intermittent",
    ],
    "other": [],
}


# =============================================================================
# Default Gases by Source Type
# =============================================================================

#: Default greenhouse gases emitted by each fugitive source type.
#: Used when facility-specific gas composition data is not available.
SOURCE_DEFAULT_GASES: Dict[str, List[str]] = {
    "wellhead": ["CH4", "CO2", "VOC"],
    "separator": ["CH4", "CO2", "VOC"],
    "dehydrator": ["CH4", "VOC"],
    "pneumatic_controller_high": ["CH4"],
    "pneumatic_controller_low": ["CH4"],
    "pneumatic_controller_intermittent": ["CH4"],
    "compressor_centrifugal": ["CH4", "CO2", "VOC"],
    "compressor_reciprocating": ["CH4", "CO2", "VOC"],
    "acid_gas_removal": ["CH4", "CO2"],
    "glycol_dehydrator": ["CH4", "VOC"],
    "pipeline_main": ["CH4"],
    "pipeline_service": ["CH4"],
    "meter_regulator": ["CH4"],
    "tank_fixed_roof": ["CH4", "VOC"],
    "tank_floating_roof": ["CH4", "VOC"],
    "coal_mine_underground": ["CH4", "CO2"],
    "coal_mine_surface": ["CH4"],
    "coal_handling": ["CH4"],
    "abandoned_mine": ["CH4", "CO2"],
    "wastewater_lagoon": ["CH4", "N2O"],
    "wastewater_digester": ["CH4"],
    "wastewater_aerobic": ["N2O"],
    "valve_gas": ["CH4", "VOC"],
    "pump_seal": ["CH4", "VOC"],
    "compressor_seal": ["CH4", "VOC"],
    "flange_connector": ["CH4", "VOC"],
}


# =============================================================================
# Data Models (16)
# =============================================================================


class FugitiveSourceInfo(BaseModel):
    """Metadata record describing a fugitive emission source type.

    Provides reference information about a source type including its
    category, applicable gases, default emission factors, and regulatory
    references. Used for source type registration and lookup by the
    FugitiveSourceDatabaseEngine.

    Attributes:
        source_type: Identifier of the fugitive emission source.
        category: Broad sector classification.
        name: Human-readable display name.
        description: Detailed description of the source and its
            emission mechanisms.
        primary_gases: List of greenhouse gas species emitted by
            this source type under normal operating conditions.
        applicable_methods: List of calculation methods that can be
            applied to this source type.
        ipcc_reference: IPCC 2006 Guidelines chapter and section.
        epa_subpart: EPA 40 CFR Part 98 subpart identifier.
        supports_ldar: Whether LDAR surveys are applicable to this source.
        supports_direct_measurement: Whether direct measurement is
            applicable to this source.
    """

    model_config = ConfigDict(frozen=True)

    source_type: FugitiveSourceType = Field(
        ...,
        description="Identifier of the fugitive emission source",
    )
    category: FugitiveSourceCategory = Field(
        ...,
        description="Broad sector classification",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable display name",
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed description of the source and its emissions",
    )
    primary_gases: List[EmissionGas] = Field(
        default_factory=list,
        description="Greenhouse gas species emitted by this source type",
    )
    applicable_methods: List[CalculationMethod] = Field(
        default_factory=list,
        description="Calculation methods applicable to this source type",
    )
    ipcc_reference: Optional[str] = Field(
        default=None,
        max_length=255,
        description="IPCC 2006 Guidelines chapter and section",
    )
    epa_subpart: Optional[str] = Field(
        default=None,
        max_length=100,
        description="EPA 40 CFR Part 98 subpart identifier",
    )
    supports_ldar: bool = Field(
        default=False,
        description="Whether LDAR surveys are applicable",
    )
    supports_direct_measurement: bool = Field(
        default=False,
        description="Whether direct measurement is applicable",
    )


class ComponentRecord(BaseModel):
    """Registration record for a single equipment component in an LDAR program.

    Tracks component-level metadata required for fugitive emission
    calculations by the average emission factor, screening ranges,
    and correlation equation methods. Each component record uniquely
    identifies a physical equipment component by its tag number within
    a facility.

    Attributes:
        component_id: Unique system identifier for this component.
        tag_number: Unique equipment tag identifier within the facility
            (e.g. V-101-FLG-001).
        component_type: Classification of the equipment component.
        service_type: Process stream service classification.
        facility_id: Identifier of the facility where the component is
            installed.
        location: Physical location description within the facility
            (e.g. process unit, area, elevation).
        installation_date: Date the component was installed or first
            registered in the LDAR program.
        leak_status: Current leak status of the component.
        screening_value_ppm: Most recent screening concentration in
            parts per million by volume (Method 21 reading).
        measured_rate_kg_hr: Most recent direct measurement rate in
            kg/hr (Hi-Flow or bagging measurement).
        operating_hours_per_year: Annual operating hours for the
            component. Default 8760 (continuous operation).
        notes: Optional notes about the component.
    """

    model_config = ConfigDict(frozen=True)

    component_id: str = Field(
        default_factory=lambda: f"comp_{uuid.uuid4().hex[:12]}",
        description="Unique system identifier for this component",
    )
    tag_number: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique equipment tag identifier within the facility",
    )
    component_type: ComponentType = Field(
        ...,
        description="Classification of the equipment component",
    )
    service_type: ServiceType = Field(
        ...,
        description="Process stream service classification",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the facility",
    )
    location: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Physical location within the facility",
    )
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Date the component was installed or registered",
    )
    leak_status: LeakStatus = Field(
        default=LeakStatus.NO_LEAK,
        description="Current leak status of the component",
    )
    screening_value_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Most recent screening concentration in ppmv",
    )
    measured_rate_kg_hr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Most recent direct measurement rate (kg/hr)",
    )
    operating_hours_per_year: Decimal = Field(
        default=Decimal("8760"),
        gt=Decimal("0"),
        le=Decimal("8784"),
        description="Annual operating hours (default 8760 = full year)",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Notes about the component",
    )


class EmissionFactorRecord(BaseModel):
    """A single emission factor record for a fugitive source-gas combination.

    Emission factors define the mass of GHG released per component-hour,
    per unit of activity, or per equipment count. Each record is scoped
    to a specific source type, greenhouse gas, source authority, and
    calculation method.

    Attributes:
        factor_id: Unique identifier for this emission factor record.
        source_type: Fugitive emission source this factor applies to.
        gas: Greenhouse gas species this factor quantifies.
        factor_value: Emission factor numeric value as a Decimal for
            precision in regulatory calculations.
        factor_unit: Unit of measurement for the factor (e.g.
            kg/hr/component, m3/day/device, tCH4/t coal).
        source: Authority that published this emission factor.
        method: Calculation method this factor is appropriate for.
        component_type: Component type this factor applies to (for
            equipment leak factors). None for non-equipment sources.
        service_type: Service type this factor applies to (for
            equipment leak factors). None for non-equipment sources.
        geography: ISO 3166 country/region code or GLOBAL.
        coal_rank: Coal rank for coal mining factors.
        wastewater_type: Wastewater type for wastewater factors.
        effective_date: Date from which this factor is valid.
        expiry_date: Date after which this factor is superseded.
        reference: Bibliographic reference or document ID.
        uncertainty_pct: Percentage uncertainty of this factor (0-100).
        notes: Optional notes about applicability or limitations.
    """

    model_config = ConfigDict(frozen=True)

    factor_id: str = Field(
        default_factory=lambda: f"fef_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this emission factor record",
    )
    source_type: FugitiveSourceType = Field(
        ...,
        description="Fugitive emission source this factor applies to",
    )
    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species this factor quantifies",
    )
    factor_value: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Emission factor numeric value",
    )
    factor_unit: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unit of measurement (e.g. kg/hr/component)",
    )
    source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="Authority that published this emission factor",
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.AVERAGE_EMISSION_FACTOR,
        description="Calculation method this factor is appropriate for",
    )
    component_type: Optional[ComponentType] = Field(
        default=None,
        description="Component type (for equipment leak factors)",
    )
    service_type: Optional[ServiceType] = Field(
        default=None,
        description="Service type (for equipment leak factors)",
    )
    geography: str = Field(
        default="GLOBAL",
        max_length=50,
        description="ISO 3166 country/region code or GLOBAL",
    )
    coal_rank: Optional[CoalRank] = Field(
        default=None,
        description="Coal rank for coal mining factors",
    )
    wastewater_type: Optional[WastewaterType] = Field(
        default=None,
        description="Wastewater type for wastewater factors",
    )
    effective_date: Optional[datetime] = Field(
        default=None,
        description="Date from which this factor is valid",
    )
    expiry_date: Optional[datetime] = Field(
        default=None,
        description="Date after which this factor is superseded",
    )
    reference: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Bibliographic reference or document ID",
    )
    uncertainty_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage uncertainty of this factor (0-100)",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Notes about applicability or limitations",
    )

    @field_validator("expiry_date")
    @classmethod
    def expiry_after_effective(
        cls, v: Optional[datetime], info: Any
    ) -> Optional[datetime]:
        """Validate that expiry_date is after effective_date when both set."""
        if v is not None and info.data.get("effective_date") is not None:
            if v <= info.data["effective_date"]:
                raise ValueError(
                    "expiry_date must be after effective_date"
                )
        return v


class SurveyRecord(BaseModel):
    """LDAR survey record for leak detection monitoring at a facility.

    Captures the metadata of a single LDAR survey event including the
    survey methodology, date, surveyor, components inspected, leaks
    found, and coverage achieved. Survey records are linked to
    individual leak records for component-level tracking.

    Attributes:
        survey_id: Unique identifier for this survey event.
        survey_type: Leak detection methodology used.
        survey_date: Date and time the survey was conducted.
        facility_id: Identifier of the facility surveyed.
        inspector_id: Identifier or name of the surveyor / inspector.
        components_surveyed: Total number of components inspected.
        leaks_detected: Number of leaks detected during the survey.
        coverage_pct: Percentage of total facility components surveyed
            (0.0-100.0). 100.0 means all registered components were
            inspected.
        total_emissions_kg_hr: Total estimated emission rate from
            all detected leaks (kg/hr).
        weather_conditions: Description of weather during the survey.
        detection_threshold_ppm: Method 21 detection threshold used
            (typically 500 or 10,000 ppmv).
        notes: Optional notes about the survey.
    """

    model_config = ConfigDict(frozen=True)

    survey_id: str = Field(
        default_factory=lambda: f"srv_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this survey event",
    )
    survey_type: SurveyType = Field(
        ...,
        description="Leak detection methodology used",
    )
    survey_date: datetime = Field(
        ...,
        description="Date and time the survey was conducted",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the facility surveyed",
    )
    inspector_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Identifier or name of the surveyor / inspector",
    )
    components_surveyed: int = Field(
        ...,
        ge=0,
        description="Total number of components inspected",
    )
    leaks_detected: int = Field(
        ...,
        ge=0,
        description="Number of leaks detected during the survey",
    )
    coverage_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of total facility components surveyed",
    )
    total_emissions_kg_hr: float = Field(
        default=0.0,
        ge=0.0,
        description="Total estimated emission rate from detected leaks",
    )
    weather_conditions: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Weather conditions during the survey",
    )
    detection_threshold_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Method 21 detection threshold (ppmv)",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Notes about the survey",
    )

    @field_validator("leaks_detected")
    @classmethod
    def leaks_not_exceed_surveyed(
        cls, v: int, info: Any
    ) -> int:
        """Validate that leaks_detected does not exceed components_surveyed."""
        components = info.data.get("components_surveyed")
        if components is not None and v > components:
            raise ValueError(
                f"leaks_detected ({v}) cannot exceed "
                f"components_surveyed ({components})"
            )
        return v


class LeakRecord(BaseModel):
    """Individual leak record for a component detected during an LDAR survey.

    Tracks the detection, screening value, status, and repair deadline
    for a single leak event. Linked to a component via component_id
    (tag number) and to a survey via the survey_id field.

    Attributes:
        leak_id: Unique identifier for this leak record.
        component_id: Tag number of the leaking component. References
            the tag_number field of a ComponentRecord.
        survey_id: Survey during which the leak was found.
        screening_value_ppm: Measured screening concentration in parts
            per million (ppmv) from Method 21 portable analyzer. None
            for OGI-only surveys that do not provide quantitative values.
        measured_rate_kg_hr: Direct measurement emission rate (kg/hr).
            None if not directly measured.
        estimated_rate_kg_hr: Estimated emission rate from emission
            factors or correlations (kg/hr).
        leak_status: Current status of the leak.
        detection_date: Date and time the leak was detected.
        repair_deadline: Regulatory deadline for completing the repair.
            Typically 15 calendar days from detection for first attempt,
            with possible delay of repair extension.
        delay_of_repair_justification: Documented justification if
            repair is delayed beyond the regulatory timeline.
        notes: Optional notes about the leak.
    """

    model_config = ConfigDict(frozen=True)

    leak_id: str = Field(
        default_factory=lambda: f"leak_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this leak record",
    )
    component_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tag number of the leaking component",
    )
    survey_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Survey during which the leak was found",
    )
    screening_value_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Measured screening concentration in ppmv",
    )
    measured_rate_kg_hr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Direct measurement emission rate (kg/hr)",
    )
    estimated_rate_kg_hr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated emission rate from EF/correlations (kg/hr)",
    )
    leak_status: LeakStatus = Field(
        default=LeakStatus.LEAK_DETECTED,
        description="Current status of the leak",
    )
    detection_date: datetime = Field(
        ...,
        description="Date and time the leak was detected",
    )
    repair_deadline: Optional[datetime] = Field(
        default=None,
        description="Regulatory deadline for completing the repair",
    )
    delay_of_repair_justification: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Justification if repair is delayed",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Notes about the leak",
    )

    @field_validator("repair_deadline")
    @classmethod
    def deadline_after_detection(
        cls, v: Optional[datetime], info: Any
    ) -> Optional[datetime]:
        """Validate that repair_deadline is after detection_date."""
        if v is not None and info.data.get("detection_date") is not None:
            if v <= info.data["detection_date"]:
                raise ValueError(
                    "repair_deadline must be after detection_date"
                )
        return v


class RepairRecord(BaseModel):
    """Record of a leak repair action performed on an equipment component.

    Documents the repair method, date, post-repair verification reading,
    and whether the repair was verified as successful. Linked to a
    LeakRecord via the leak_id field.

    Attributes:
        repair_id: Unique identifier for this repair record.
        leak_id: Identifier of the leak record being repaired.
        component_id: Tag number of the component that was repaired.
        repair_date: Date and time the repair was completed.
        repair_method: Description of the repair technique applied
            (e.g. tightening, packing replacement, valve replacement,
            gasket replacement, bonnet re-torque).
        pre_repair_rate_kg_hr: Emission rate before repair (kg/hr).
        post_repair_ppm: Post-repair screening value in ppmv.
            Must be below the LDAR threshold to confirm successful
            repair.
        post_repair_rate_kg_hr: Emission rate after repair (kg/hr).
        emissions_reduced_kg_hr: Emission reduction achieved (kg/hr).
        repair_cost_usd: Cost of the repair in US dollars.
        is_verified: Whether the repair has been verified by a
            follow-up survey confirming the leak is resolved.
        notes: Optional notes about the repair.
    """

    model_config = ConfigDict(frozen=True)

    repair_id: str = Field(
        default_factory=lambda: f"repr_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this repair record",
    )
    leak_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the leak record being repaired",
    )
    component_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Tag number of the component that was repaired",
    )
    repair_date: datetime = Field(
        ...,
        description="Date and time the repair was completed",
    )
    repair_method: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Description of the repair technique applied",
    )
    pre_repair_rate_kg_hr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Emission rate before repair (kg/hr)",
    )
    post_repair_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Post-repair screening value in ppmv",
    )
    post_repair_rate_kg_hr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Emission rate after repair (kg/hr)",
    )
    emissions_reduced_kg_hr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Emission reduction achieved (kg/hr)",
    )
    repair_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Cost of the repair in US dollars",
    )
    is_verified: bool = Field(
        default=False,
        description="Whether the repair has been verified by follow-up survey",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Notes about the repair",
    )


class CalculationRequest(BaseModel):
    """Input data for a single fugitive emission calculation.

    Represents one calculation request for a specific fugitive emission
    source, time period, and methodology. The calculation engine uses
    this input together with emission factors, component counts, and
    gas composition data to compute GHG emissions.

    Attributes:
        source_type: Fugitive emission source type to calculate.
        calculation_method: Calculation methodology to apply.
        activity_data: Activity data quantity (e.g. number of components,
            tonnes of coal produced, m3 of wastewater treated, number
            of pneumatic devices, operating hours).
        activity_unit: Unit of the activity data (e.g. count, tonnes,
            m3, hours).
        component_counts: Optional dictionary of component counts keyed
            by component_type and service_type tuple string
            (e.g. "valve:gas" -> 150). Required for average EF method.
        gas_composition_ch4: Methane mole fraction in the process gas
            stream (0.0-1.0). Used to convert total organic compound
            emission factors to CH4-specific emissions. Default 0.80
            for natural gas.
        gas_composition_voc: VOC mole fraction in the process gas
            stream (0.0-1.0). Used to convert total organic compound
            emission factors to VOC-specific emissions.
        gwp_source: IPCC AR edition for GWP values.
        ef_source: Source authority for emission factors.
        operating_hours: Annual operating hours of the equipment or
            facility. Default 8760 (full year). Used with hourly
            emission factors to calculate annual totals.
        recovery_rate: Fraction of emissions captured or recovered
            (0.0-1.0). Used for pneumatic devices with gas recovery,
            tanks with vapor recovery units, or mines with methane
            drainage. Default 0.0 (no recovery).
        coal_rank: Coal rank for coal mining calculations.
        wastewater_type: Wastewater type for wastewater calculations.
        tank_type: Tank type for tank storage calculations.
        geography: Optional ISO 3166 code for region-specific factors.
        facility_id: Optional facility identifier for aggregation.
        period_start: Optional start of the reporting period.
        period_end: Optional end of the reporting period.
        notes: Optional notes for the calculation record.
    """

    model_config = ConfigDict(frozen=True)

    source_type: FugitiveSourceType = Field(
        ...,
        description="Fugitive emission source type to calculate",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.AVERAGE_EMISSION_FACTOR,
        description="Calculation methodology to apply",
    )
    activity_data: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Activity data quantity",
    )
    activity_unit: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unit of the activity data (e.g. count, tonnes, m3)",
    )
    component_counts: Optional[Dict[str, int]] = Field(
        default=None,
        description=(
            "Component counts keyed by 'component_type:service_type' "
            "(e.g. 'valve:gas' -> 150)"
        ),
    )
    gas_composition_ch4: Decimal = Field(
        default=Decimal("0.80"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Methane mole fraction in process gas stream (0.0-1.0)",
    )
    gas_composition_voc: Decimal = Field(
        default=Decimal("0.10"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="VOC mole fraction in process gas stream (0.0-1.0)",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC AR edition for GWP values",
    )
    ef_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="Source authority for emission factors",
    )
    operating_hours: Decimal = Field(
        default=Decimal("8760"),
        gt=Decimal("0"),
        le=Decimal("8784"),
        description="Annual operating hours (default 8760 = full year)",
    )
    recovery_rate: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Fraction of emissions captured or recovered (0.0-1.0)",
    )
    coal_rank: Optional[CoalRank] = Field(
        default=None,
        description="Coal rank for coal mining calculations",
    )
    wastewater_type: Optional[WastewaterType] = Field(
        default=None,
        description="Wastewater type for wastewater calculations",
    )
    tank_type: Optional[TankType] = Field(
        default=None,
        description="Tank type for tank storage calculations",
    )
    geography: Optional[str] = Field(
        default=None,
        max_length=50,
        description="ISO 3166 code for region-specific factors",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Facility identifier for aggregation",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the reporting period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the reporting period",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Notes for the calculation record",
    )

    @field_validator("period_end")
    @classmethod
    def period_end_after_start(
        cls, v: Optional[datetime], info: Any
    ) -> Optional[datetime]:
        """Validate that period_end is after period_start when both set."""
        if v is not None and info.data.get("period_start") is not None:
            if v <= info.data["period_start"]:
                raise ValueError(
                    "period_end must be after period_start"
                )
        return v

    @field_validator("gas_composition_voc")
    @classmethod
    def validate_total_gas_fraction(
        cls, v: Decimal, info: Any
    ) -> Decimal:
        """Validate CH4 + VOC fractions do not exceed 1.0."""
        ch4 = info.data.get("gas_composition_ch4", Decimal("0.0"))
        if ch4 + v > Decimal("1.0"):
            raise ValueError(
                f"gas_composition_ch4 ({ch4}) + "
                f"gas_composition_voc ({v}) must not exceed 1.0"
            )
        return v


class CalculationResult(BaseModel):
    """Complete result of a single fugitive emission calculation.

    Contains all calculated emissions by gas, total CO2e, the methodology
    parameters used, and a SHA-256 provenance hash for audit trail
    integrity. Emissions are reported in kilograms for precision and
    can be converted to tonnes for reporting.

    Attributes:
        calculation_id: Unique identifier for this calculation result.
        source_type: Fugitive emission source type calculated.
        method: Calculation methodology applied.
        total_co2e_kg: Total CO2-equivalent emissions in kilograms.
        ch4_kg: Methane emissions in kilograms.
        co2_kg: Carbon dioxide emissions in kilograms.
        n2o_kg: Nitrous oxide emissions in kilograms.
        voc_kg: Volatile organic compound emissions in kilograms.
            VOC is not included in total_co2e_kg.
        total_co2e_tonnes: Total CO2-equivalent emissions in tonnes.
        component_count: Number of components included in calculation.
        leaks_detected: Number of leaks detected (LDAR-based methods).
        leak_rate_pct: Leak rate percentage (LDAR-based methods).
        emissions_before_repair_kg: Emissions before repairs (kg).
        emissions_after_repair_kg: Emissions after repairs (kg).
        emission_reduction_kg: Emission reduction from repairs (kg).
        uncertainty_pct: Estimated uncertainty as a percentage (0-100).
            Based on the calculation method uncertainty tier.
        provenance_hash: SHA-256 hash of all inputs, parameters, and
            intermediate values for complete audit trail integrity.
        calculation_trace: Ordered list of human-readable calculation steps.
        processing_time_ms: Time taken to perform the calculation in
            milliseconds.
        timestamp: UTC timestamp when the calculation was performed.
        facility_id: Facility identifier (if provided).
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        gwp_source: GWP source used.
        ef_source: Emission factor source used.
    """

    model_config = ConfigDict(frozen=True)

    calculation_id: str = Field(
        default_factory=lambda: f"fecalc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this calculation result",
    )
    source_type: FugitiveSourceType = Field(
        ...,
        description="Fugitive emission source type calculated",
    )
    method: CalculationMethod = Field(
        ...,
        description="Calculation methodology applied",
    )
    total_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2-equivalent emissions in kilograms",
    )
    ch4_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Methane emissions in kilograms",
    )
    co2_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Carbon dioxide emissions in kilograms",
    )
    n2o_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Nitrous oxide emissions in kilograms",
    )
    voc_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Volatile organic compound emissions in kilograms",
    )
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total CO2-equivalent emissions in tonnes",
    )
    component_count: int = Field(
        default=0,
        ge=0,
        description="Number of components included in calculation",
    )
    leaks_detected: int = Field(
        default=0,
        ge=0,
        description="Number of leaks detected (LDAR-based methods)",
    )
    leak_rate_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Leak rate percentage (LDAR-based methods)",
    )
    emissions_before_repair_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions before repairs (kg)",
    )
    emissions_after_repair_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions after repairs (kg)",
    )
    emission_reduction_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emission reduction from repairs (kg)",
    )
    uncertainty_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Estimated uncertainty as a percentage (0-100)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail integrity",
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        max_length=MAX_TRACE_STEPS,
        description="Ordered list of human-readable calculation steps",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Calculation processing time in milliseconds",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the calculation was performed",
    )
    facility_id: Optional[str] = Field(
        default=None,
        description="Facility identifier",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the reporting period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the reporting period",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source used",
    )
    ef_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="Emission factor source used",
    )


class CalculationDetailResult(BaseModel):
    """Detailed per-gas calculation breakdown for audit trail traceability.

    Captures the emission factor, raw emission mass, GWP multiplier,
    and CO2-equivalent for a single greenhouse gas within a fugitive
    emission calculation. Multiple CalculationDetailResult records
    compose the full breakdown of a CalculationResult.

    Attributes:
        gas: Greenhouse gas species for this detail line.
        emission_factor: Emission factor value applied for this gas.
        emission_factor_unit: Unit of the emission factor.
        emission_factor_source: Source authority for the factor.
        raw_emissions_kg: Calculated raw emissions in kilograms before
            GWP conversion.
        gwp_value: Global Warming Potential multiplier applied. Set to
            Decimal("0") for VOC where no standard GWP exists.
        co2e_kg: Calculated CO2-equivalent emissions in kilograms.
        calculation_trace: Step-by-step calculation trace for this gas.
    """

    model_config = ConfigDict(frozen=True)

    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species for this detail line",
    )
    emission_factor: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor value applied for this gas",
    )
    emission_factor_unit: str = Field(
        default="",
        description="Unit of the emission factor",
    )
    emission_factor_source: str = Field(
        default="",
        description="Source authority for the emission factor",
    )
    raw_emissions_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Raw emissions in kilograms before GWP conversion",
    )
    gwp_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="GWP multiplier applied (0 for VOC)",
    )
    co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2-equivalent emissions in kilograms",
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step calculation trace for this gas",
    )


class BatchCalculationRequest(BaseModel):
    """Request model for batch fugitive emission calculations.

    Groups multiple calculation inputs for processing as a single
    batch, sharing common parameters like GWP source and compliance
    framework preferences. Designed for facility-level inventories
    where many source types must be calculated together.

    Attributes:
        calculations: List of individual calculation requests.
        gwp_source: IPCC AR edition for GWP values (shared by batch).
        enable_compliance: Whether to run compliance checks on results.
        compliance_frameworks: Frameworks to check against.
        organization_id: Organization identifier for aggregation.
        reporting_period: Temporal granularity for the batch.
    """

    model_config = ConfigDict(frozen=True)

    calculations: List[CalculationRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_CALCULATIONS_PER_BATCH,
        description="List of individual calculation requests",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC AR edition for GWP values",
    )
    enable_compliance: bool = Field(
        default=False,
        description="Whether to run compliance checks on results",
    )
    compliance_frameworks: List[str] = Field(
        default_factory=list,
        description="Regulatory frameworks to check against",
    )
    organization_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Organization identifier for aggregation",
    )
    reporting_period: Optional[ReportingPeriod] = Field(
        default=None,
        description="Temporal granularity for the batch",
    )


class BatchCalculationResult(BaseModel):
    """Response model for a batch fugitive emission calculation.

    Aggregates individual calculation results with batch-level totals,
    emissions breakdown by source type and gas, and processing metadata.

    Attributes:
        success: Whether all calculations in the batch succeeded.
        total: Total number of calculation requests in the batch.
        successful: Number of calculations that completed successfully.
        failed: Number of calculations that failed.
        total_co2e_kg: Batch total CO2-equivalent emissions in kilograms.
        total_co2e_tonnes: Batch total CO2-equivalent in metric tonnes.
        emissions_by_source_type: Emissions by source type (kg CO2e).
        emissions_by_gas: Emissions by gas species (kg).
        total_components: Total components across all calculations.
        total_leaks: Total leaks detected across all calculations.
        total_emission_reduction_kg: Total reduction from repairs (kg).
        results: List of individual calculation results (successful only).
        processing_time_ms: Total batch processing time in ms.
        provenance_hash: SHA-256 hash covering the entire batch.
        gwp_source: GWP source used for this batch.
        compliance_results: Compliance check results (if requested).
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(
        ...,
        description="Whether all calculations in the batch succeeded",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of calculation requests in the batch",
    )
    successful: int = Field(
        default=0,
        ge=0,
        description="Number of calculations that completed successfully",
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Number of calculations that failed",
    )
    total_co2e_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Batch total CO2-equivalent emissions in kilograms",
    )
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Batch total CO2-equivalent in metric tonnes",
    )
    emissions_by_source_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by source type (kg CO2e)",
    )
    emissions_by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by gas species (kg)",
    )
    total_components: int = Field(
        default=0,
        ge=0,
        description="Total components across all calculations",
    )
    total_leaks: int = Field(
        default=0,
        ge=0,
        description="Total leaks detected across all calculations",
    )
    total_emission_reduction_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total emission reduction from repairs (kg)",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total batch processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash covering the entire batch result",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source used for this batch",
    )
    compliance_results: List["ComplianceCheckResult"] = Field(
        default_factory=list,
        description="Compliance check results (if requested)",
    )


class UncertaintyRequest(BaseModel):
    """Request model for uncertainty quantification of a fugitive emission.

    Configures Monte Carlo simulation parameters for propagating
    emission factor, activity data, and gas composition uncertainties
    through the calculation to produce confidence intervals and a
    data quality indicator score.

    Attributes:
        calculation_data: The base calculation request to quantify.
        method: Uncertainty quantification method (currently only
            Monte Carlo is supported; reserved for future analytical
            methods).
        iterations: Number of Monte Carlo simulation iterations.
            Higher values yield more precise confidence intervals.
        seed: Random seed for reproducibility. Set to 0 for
            non-deterministic runs.
        confidence_levels: Confidence levels for interval calculation.
        include_contributions: Whether to compute parameter
            contribution analysis.
    """

    model_config = ConfigDict(frozen=True)

    calculation_data: CalculationRequest = Field(
        ...,
        description="The base calculation request to quantify",
    )
    method: str = Field(
        default="monte_carlo",
        description="Uncertainty quantification method",
    )
    iterations: int = Field(
        default=5000,
        gt=0,
        le=1_000_000,
        description="Number of Monte Carlo iterations",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility (0 = non-deterministic)",
    )
    confidence_levels: List[float] = Field(
        default_factory=lambda: [90.0, 95.0, 99.0],
        description="Confidence levels for interval calculation",
    )
    include_contributions: bool = Field(
        default=True,
        description="Whether to compute parameter contribution analysis",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate and normalize the uncertainty method."""
        normalised = v.strip().lower()
        valid = {"monte_carlo", "analytical"}
        if normalised not in valid:
            raise ValueError(
                f"method must be one of {sorted(valid)}, got '{v}'"
            )
        return normalised

    @field_validator("confidence_levels")
    @classmethod
    def validate_confidence_levels(cls, v: List[float]) -> List[float]:
        """Validate all confidence levels are in (0, 100)."""
        for lvl in v:
            if not (0.0 < lvl < 100.0):
                raise ValueError(
                    f"Each confidence level must be in (0, 100), got {lvl}"
                )
        return v


class UncertaintyResult(BaseModel):
    """Monte Carlo uncertainty quantification result for a fugitive emission.

    Provides statistical characterisation of emission estimate uncertainty
    including mean, standard deviation, confidence intervals, and a
    data quality indicator (DQI) score on a 1-5 scale.

    Attributes:
        success: Whether the uncertainty analysis completed successfully.
        method: Uncertainty quantification method used.
        mean_co2e_kg: Mean CO2-equivalent emission estimate (kilograms).
        std_dev_kg: Standard deviation (kilograms).
        coefficient_of_variation: CV = std_dev / mean (dimensionless).
        confidence_intervals: Confidence intervals keyed by level
            string (e.g. "90" -> (lower, upper) in kg CO2e).
        iterations: Number of Monte Carlo iterations performed.
        seed_used: Random seed used for reproducibility.
        dqi_score: Data Quality Indicator score on a 1-5 scale.
            1 = highest quality (direct measurement, facility-specific).
            5 = lowest quality (default factors, global average).
        contributions: Parameter contribution to total variance, keyed
            by parameter name (e.g. "emission_factor" -> 0.45).
        source_type: Source type analyzed.
        calculation_method: Calculation method used.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(
        ...,
        description="Whether the uncertainty analysis completed",
    )
    method: str = Field(
        ...,
        description="Uncertainty quantification method used",
    )
    mean_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Mean CO2-equivalent emission estimate (kg)",
    )
    std_dev_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Standard deviation (kg)",
    )
    coefficient_of_variation: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CV = std_dev / mean (dimensionless)",
    )
    confidence_intervals: Dict[str, Tuple[Decimal, Decimal]] = Field(
        default_factory=dict,
        description=(
            "Confidence intervals keyed by level "
            "(e.g. '90' -> (lower, upper) in kg CO2e)"
        ),
    )
    iterations: int = Field(
        default=5000,
        gt=0,
        description="Number of Monte Carlo iterations performed",
    )
    seed_used: int = Field(
        default=42,
        ge=0,
        description="Random seed used for reproducibility",
    )
    dqi_score: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("1"),
        le=Decimal("5"),
        description="Data Quality Indicator score (1-5 scale)",
    )
    contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameter contribution to total variance",
    )
    source_type: Optional[FugitiveSourceType] = Field(
        default=None,
        description="Source type analyzed",
    )
    calculation_method: Optional[CalculationMethod] = Field(
        default=None,
        description="Calculation method used",
    )


class ComplianceCheckResult(BaseModel):
    """Result of regulatory compliance checks across multiple frameworks.

    Captures the assessment of whether a fugitive emission calculation
    meets the requirements of applicable regulatory frameworks, with
    per-framework status, findings count, and detailed results.

    Attributes:
        success: Whether the compliance check process completed.
        frameworks_checked: List of regulatory framework names checked.
        compliant: Number of frameworks with COMPLIANT status.
        non_compliant: Number of frameworks with NON_COMPLIANT status.
        partial: Number of frameworks with PARTIAL status.
        results: Detailed compliance results keyed by framework name.
            Each entry contains status, findings list, and recommendations.
        checked_at: UTC timestamp when the check was performed.
        calculation_id: The calculation this check applies to.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(
        ...,
        description="Whether the compliance check process completed",
    )
    frameworks_checked: List[str] = Field(
        default_factory=list,
        description="Regulatory framework names checked",
    )
    compliant: int = Field(
        default=0,
        ge=0,
        description="Number of frameworks with COMPLIANT status",
    )
    non_compliant: int = Field(
        default=0,
        ge=0,
        description="Number of frameworks with NON_COMPLIANT status",
    )
    partial: int = Field(
        default=0,
        ge=0,
        description="Number of frameworks with PARTIAL status",
    )
    results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Detailed compliance results keyed by framework name, "
            "each containing status, findings, and recommendations"
        ),
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the check was performed",
    )
    calculation_id: Optional[str] = Field(
        default=None,
        description="Calculation this check applies to",
    )


class AggregationRequest(BaseModel):
    """Request model for aggregating fugitive emissions.

    Defines the scope and grouping parameters for rolling up individual
    calculation results into aggregate totals by facility, source type,
    source category, or gas species.

    Attributes:
        calculation_ids: List of calculation result IDs to aggregate.
            At least one ID is required.
        group_by: Grouping dimensions for the aggregation. Valid values
            are: source_type, category, facility, gas, method,
            coal_rank, wastewater_type, tank_type.
        facility_id: Optional facility filter.
        organization_id: Optional organization filter.
        period_start: Optional start of the aggregation period.
        period_end: Optional end of the aggregation period.
        reporting_period: Temporal granularity for the aggregation.
        include_ldar_summary: Whether to include LDAR summary in output.
    """

    model_config = ConfigDict(frozen=True)

    calculation_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of calculation result IDs to aggregate",
    )
    group_by: List[str] = Field(
        default_factory=lambda: ["source_type"],
        description="Grouping dimensions for the aggregation",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Facility filter",
    )
    organization_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Organization filter",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the aggregation period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the aggregation period",
    )
    reporting_period: Optional[ReportingPeriod] = Field(
        default=None,
        description="Temporal granularity for the aggregation",
    )
    include_ldar_summary: bool = Field(
        default=True,
        description="Whether to include LDAR summary in output",
    )

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, v: List[str]) -> List[str]:
        """Validate group_by dimensions are recognized."""
        valid_dims = {
            "source_type",
            "category",
            "facility",
            "gas",
            "method",
            "coal_rank",
            "wastewater_type",
            "tank_type",
        }
        for dim in v:
            if dim not in valid_dims:
                raise ValueError(
                    f"group_by dimension '{dim}' not recognized; "
                    f"valid dimensions: {sorted(valid_dims)}"
                )
        return v

    @field_validator("period_end")
    @classmethod
    def agg_end_after_start(
        cls, v: Optional[datetime], info: Any
    ) -> Optional[datetime]:
        """Validate that period_end is after period_start when both set."""
        if v is not None and info.data.get("period_start") is not None:
            if v <= info.data["period_start"]:
                raise ValueError(
                    "period_end must be after period_start"
                )
        return v


class AggregationResult(BaseModel):
    """Result of a fugitive emission aggregation across calculations.

    Rolls up individual calculation results into aggregate totals with
    breakdowns by the requested grouping dimensions (source type, gas).

    Attributes:
        total_co2e_kg: Aggregate total CO2-equivalent in kilograms.
        total_co2e_tonnes: Aggregate total CO2-equivalent in tonnes.
        by_source_type: Emissions keyed by source type value (kg CO2e).
        by_gas: Emissions keyed by gas species (kg).
        by_group: Emissions keyed by grouping dimension values.
        total_components: Aggregate total components.
        total_leaks: Aggregate total leaks detected.
        total_emission_reduction_kg: Aggregate emission reduction (kg).
        calculation_count: Number of calculations aggregated.
        source_types_included: Distinct source types in aggregation.
        categories_included: Distinct source categories in aggregation.
        ldar_summary: LDAR summary data (if requested).
        provenance_hash: SHA-256 hash for audit trail integrity.
        facility_id: Facility filter applied (if any).
        organization_id: Organization filter applied (if any).
        period_start: Start of the aggregation period (if any).
        period_end: End of the aggregation period (if any).
        reporting_period: Temporal granularity (if any).
    """

    model_config = ConfigDict(frozen=True)

    total_co2e_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate total CO2-equivalent in kilograms",
    )
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate total CO2-equivalent in tonnes",
    )
    by_source_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions keyed by source type (kg CO2e)",
    )
    by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions keyed by gas species (kg)",
    )
    by_group: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description="Emissions keyed by grouping dimension values",
    )
    total_components: int = Field(
        default=0,
        ge=0,
        description="Aggregate total components",
    )
    total_leaks: int = Field(
        default=0,
        ge=0,
        description="Aggregate total leaks detected",
    )
    total_emission_reduction_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate emission reduction from repairs (kg)",
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations aggregated",
    )
    source_types_included: List[str] = Field(
        default_factory=list,
        description="Distinct source types in aggregation",
    )
    categories_included: List[str] = Field(
        default_factory=list,
        description="Distinct source categories in aggregation",
    )
    ldar_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LDAR summary data (if requested)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail integrity",
    )
    facility_id: Optional[str] = Field(
        default=None,
        description="Facility filter applied",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization filter applied",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the aggregation period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the aggregation period",
    )
    reporting_period: Optional[ReportingPeriod] = Field(
        default=None,
        description="Temporal granularity",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_COMPONENTS_PER_CALC",
    "MAX_SURVEYS_PER_FACILITY",
    "MAX_LEAKS_PER_SURVEY",
    "DEFAULT_LEAK_THRESHOLD_PPM",
    "DEFAULT_REPAIR_DEADLINE_DAYS",
    "MAX_DELAY_OF_REPAIR_DAYS",
    "GWP_VALUES",
    "EPA_COMPONENT_EMISSION_FACTORS",
    "EPA_CORRELATION_COEFFICIENTS",
    "IPCC_COAL_EMISSION_FACTORS",
    "COAL_METHANE_FACTORS",
    "WASTEWATER_MCF",
    "PNEUMATIC_RATES_M3_PER_DAY",
    "SOURCE_CATEGORY_MAP",
    "SOURCE_DEFAULT_GASES",
    # Enums (16)
    "FugitiveSourceCategory",
    "FugitiveSourceType",
    "ComponentType",
    "ServiceType",
    "EmissionGas",
    "CalculationMethod",
    "EmissionFactorSource",
    "GWPSource",
    "SurveyType",
    "LeakStatus",
    "CoalRank",
    "WastewaterType",
    "ComplianceStatus",
    "ReportingPeriod",
    "UnitType",
    "TankType",
    # Data Models (16)
    "FugitiveSourceInfo",
    "ComponentRecord",
    "EmissionFactorRecord",
    "SurveyRecord",
    "LeakRecord",
    "RepairRecord",
    "CalculationRequest",
    "CalculationResult",
    "CalculationDetailResult",
    "BatchCalculationRequest",
    "BatchCalculationResult",
    "UncertaintyRequest",
    "UncertaintyResult",
    "ComplianceCheckResult",
    "AggregationRequest",
    "AggregationResult",
]
