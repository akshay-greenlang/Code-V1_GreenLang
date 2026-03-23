# -*- coding: utf-8 -*-
"""
Waste Treatment Emissions Agent Data Models - AGENT-MRV-007

Pydantic v2 data models for the Waste Treatment Emissions Agent SDK
covering GHG Protocol Scope 1 on-site waste treatment emission
calculations including:
- 19 IPCC waste categories (MSW, industrial, C&D, organic, food, yard,
  paper, cardboard, plastic, metal, glass, textiles, wood, rubber,
  e-waste, hazardous, medical, sludge, mixed)
- 15 treatment methods (landfill, incineration, composting, AD, MBT,
  pyrolysis, gasification, chemical, thermal, biological, open burning,
  open dumping, recycling, landfill gas capture, incineration with
  energy recovery)
- 7 calculation methods (IPCC FOD, Tier 1/2/3, mass balance, direct
  measurement, spend-based)
- Biological treatment engine (composting, AD, MBT, vermicomposting)
- Thermal treatment engine (incineration, pyrolysis, gasification,
  open burning with fossil/biogenic CO2 separation)
- Wastewater treatment engine (BOD/COD-based CH4, N2O from
  nitrification/denitrification)
- Methane recovery and utilization tracking
- Energy recovery and grid displacement offset credits
- Monte Carlo uncertainty quantification
- Multi-framework regulatory compliance (IPCC, GHG Protocol, CSRD,
  EPA, EU IED, DEFRA, ISO 14064)
- SHA-256 provenance chain for complete audit trails

Enumerations (16):
    - WasteCategory, TreatmentMethod, CompostingType, IncineratorType,
      WastewaterSystem, CalculationMethod, EmissionGas, GWPSource,
      EmissionFactorSource, DataQualityTier, FacilityType,
      BiogasComponent, ClimateZone, ComplianceStatus, ReportingPeriod,
      EmissionScope

Constants:
    - GWP_VALUES: IPCC AR4/AR5/AR6/AR6_20YR GWP values (Decimal)
    - CONVERSION_FACTOR_CO2_C: 44/12 molecular weight ratio
    - CH4_C_RATIO: 16/12 molecular weight ratio
    - N2O_N_RATIO: 44/28 molecular weight ratio
    - CH4_DENSITY_STP: CH4 density at STP (tonnes/m3)
    - IPCC_DOC_VALUES: Degradable organic carbon fraction by waste type
    - IPCC_MCF_VALUES: Methane correction factors by landfill type
    - IPCC_CARBON_CONTENT: Carbon content and fossil fraction by waste
    - IPCC_COMPOSTING_EF: Composting/AD/MBT emission factors
    - IPCC_INCINERATION_EF: Incineration EFs by incinerator type
    - IPCC_WASTEWATER_MCF: MCF values for wastewater systems
    - INCINERATION_NCV: Net calorific values by waste type
    - HALF_LIFE_VALUES: Waste decomposition half-lives by climate/waste

Data Models (18):
    - WasteComposition, TreatmentFacilityInfo, WasteStreamInfo,
      EmissionFactorRecord, CalculationRequest, GasEmissionDetail,
      CalculationResult, BatchCalculationRequest, BatchCalculationResult,
      BiologicalTreatmentInput, ThermalTreatmentInput,
      WastewaterTreatmentInput, MethaneRecoveryRecord,
      EnergyRecoveryRecord, ComplianceCheckResult, UncertaintyRequest,
      UncertaintyResult, AggregationRequest, AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 Waste Treatment Emissions (GL-MRV-SCOPE1-007)
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

#: Maximum waste streams per single calculation request.
MAX_STREAMS_PER_CALC: int = 50

#: Maximum facilities per tenant.
MAX_FACILITIES_PER_TENANT: int = 10_000

#: Default IPCC DOCf (fraction of DOC that decomposes).
DEFAULT_DOCf: Decimal = Decimal("0.5")

#: Default fraction of CH4 in landfill gas.
DEFAULT_F_CH4_LFG: Decimal = Decimal("0.5")

#: Default oxidation factor for modern incinerators.
DEFAULT_OXIDATION_FACTOR: Decimal = Decimal("1.0")

#: Default oxidation factor for open burning (incomplete combustion).
DEFAULT_OPEN_BURN_OXIDATION: Decimal = Decimal("0.58")

#: Default flare destruction efficiency.
DEFAULT_FLARE_DESTRUCTION_EFF: Decimal = Decimal("0.98")


# =============================================================================
# Enumerations (16)
# =============================================================================


class WasteCategory(str, Enum):
    """IPCC waste categories for treatment emission calculations.

    Nineteen waste categories as defined in IPCC 2006 Guidelines
    Volume 5 (Waste) and 2019 Refinement. Each waste stream entering
    a treatment facility is classified into exactly one category to
    determine applicable emission factors, degradable organic carbon
    (DOC) fractions, fossil carbon fractions, and net calorific values.

    MSW: Municipal solid waste - mixed residential and commercial
        waste collected by or on behalf of municipalities.
    INDUSTRIAL: Industrial waste from manufacturing and production
        processes, excluding hazardous waste.
    CONSTRUCTION_DEMOLITION: Construction and demolition waste
        including concrete, bricks, tiles, ceramics, and wood.
    ORGANIC: Biodegradable organic fraction of waste streams,
        typically separated at source or by MBT.
    FOOD: Pre-consumer and post-consumer food waste including
        kitchen scraps, spoiled food, and food processing residues.
    YARD: Yard and garden waste including leaves, grass clippings,
        branches, and tree trimmings.
    PAPER: Office paper, newspaper, magazines, printing paper,
        and other cellulose-based paper products.
    CARDBOARD: Corrugated cardboard, flat cardboard, and
        paperboard packaging.
    PLASTIC: All polymer types including PE, PP, PET, PS, PVC,
        and mixed plastics from packaging and products.
    METAL: Ferrous metals (steel, iron) and non-ferrous metals
        (aluminum, copper) from packaging and products.
    GLASS: Container glass (bottles, jars) and flat glass from
        windows and other applications.
    TEXTILES: Natural fibers (cotton, wool, silk) and synthetic
        fibers (polyester, nylon, acrylic) from clothing and furnishings.
    WOOD: Treated and untreated wood waste including furniture,
        pallets, construction timber, and wood packaging.
    RUBBER: Tires, rubber products, and elastomeric materials
        from automotive and industrial applications.
    E_WASTE: Waste electrical and electronic equipment (WEEE)
        including computers, phones, and household appliances.
    HAZARDOUS: Chemical, biological, or radiological waste
        requiring special treatment and disposal procedures.
    MEDICAL: Clinical waste from healthcare facilities including
        sharps, infectious waste, and pharmaceutical waste.
    SLUDGE: Sewage sludge from municipal wastewater treatment
        and industrial sludge from process wastewater.
    MIXED: Unsorted or commingled waste that has not been
        separated into individual waste categories.
    """

    MSW = "msw"
    INDUSTRIAL = "industrial"
    CONSTRUCTION_DEMOLITION = "construction_demolition"
    ORGANIC = "organic"
    FOOD = "food"
    YARD = "yard"
    PAPER = "paper"
    CARDBOARD = "cardboard"
    PLASTIC = "plastic"
    METAL = "metal"
    GLASS = "glass"
    TEXTILES = "textiles"
    WOOD = "wood"
    RUBBER = "rubber"
    E_WASTE = "e_waste"
    HAZARDOUS = "hazardous"
    MEDICAL = "medical"
    SLUDGE = "sludge"
    MIXED = "mixed"


class TreatmentMethod(str, Enum):
    """Waste treatment methods for emission calculation.

    Fifteen treatment methods covering biological, thermal, chemical,
    and disposal pathways as defined in IPCC 2006 Guidelines Volume 5
    and 2019 Refinement. The treatment method determines the applicable
    emission factors, calculation methodology, and emission gas profile.

    LANDFILL: Managed anaerobic disposal in engineered landfills.
        Primary emission is CH4 via first-order decay.
    LANDFILL_GAS_CAPTURE: Managed landfill with landfill gas (LFG)
        collection system for flaring or energy recovery.
    INCINERATION: Mass burn incineration without energy recovery.
        Emissions include fossil CO2, CH4, N2O, and CO.
    INCINERATION_ENERGY_RECOVERY: Waste-to-energy (WtE) plants
        with electricity and/or heat recovery, providing offset credits.
    RECYCLING: Material recovery and reprocessing. Avoided emissions
        from displacing virgin material production.
    COMPOSTING: Aerobic decomposition of organic waste producing
        compost. Emissions include CH4 and N2O.
    ANAEROBIC_DIGESTION: Biogas production from organic waste under
        anaerobic conditions. CH4 emissions from leaks/venting.
    MBT: Mechanical-biological treatment combining mechanical
        sorting with biological stabilization.
    PYROLYSIS: Thermal decomposition of waste in absence of oxygen
        producing syngas, bio-oil, and char.
    GASIFICATION: Partial oxidation of waste at high temperature
        producing synthesis gas (syngas).
    CHEMICAL_TREATMENT: Chemical processes including neutralization,
        oxidation, reduction, and precipitation.
    THERMAL_TREATMENT: Other thermal processes including autoclaving,
        microwave treatment, and plasma arc.
    BIOLOGICAL_TREATMENT: Bioaugmentation, bioremediation, and other
        biological degradation processes.
    OPEN_BURNING: Uncontrolled combustion of waste in open air.
        Common in developing regions. High emission factors.
    OPEN_DUMPING: Unmanaged disposal without engineering controls.
        Low MCF, significant fugitive CH4 emissions.
    """

    LANDFILL = "landfill"
    LANDFILL_GAS_CAPTURE = "landfill_gas_capture"
    INCINERATION = "incineration"
    INCINERATION_ENERGY_RECOVERY = "incineration_energy_recovery"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    MBT = "mbt"
    PYROLYSIS = "pyrolysis"
    GASIFICATION = "gasification"
    CHEMICAL_TREATMENT = "chemical_treatment"
    THERMAL_TREATMENT = "thermal_treatment"
    BIOLOGICAL_TREATMENT = "biological_treatment"
    OPEN_BURNING = "open_burning"
    OPEN_DUMPING = "open_dumping"


class CompostingType(str, Enum):
    """Types of composting systems for biological treatment.

    Composting type affects CH4 and N2O emission factors due to
    differences in aeration, temperature control, and moisture
    management.

    WINDROW: Open windrow composting with periodic turning.
        Moderate aeration, higher CH4 risk in anaerobic pockets.
    IN_VESSEL: Enclosed composting in rotating drums, tunnels,
        or agitated bays with controlled aeration and temperature.
    AERATED_STATIC_PILE: Static pile with forced aeration through
        embedded perforated pipes. Good oxygen distribution.
    VERMICOMPOSTING: Composting using earthworms (Eisenia fetida)
        for decomposition. Low temperature, low emissions.
    HOME_COMPOSTING: Small-scale backyard composting by households.
        Variable management quality, limited monitoring.
    """

    WINDROW = "windrow"
    IN_VESSEL = "in_vessel"
    AERATED_STATIC_PILE = "aerated_static_pile"
    VERMICOMPOSTING = "vermicomposting"
    HOME_COMPOSTING = "home_composting"


class IncineratorType(str, Enum):
    """Types of incinerator technology for thermal treatment.

    Incinerator type determines N2O and CH4 emission factors per
    IPCC 2006 Guidelines Volume 5 Table 5.3. Modern continuous
    incinerators have lower CH4 due to higher combustion efficiency.

    STOKER_GRATE: Conventional stoker grate incinerator for mass
        burn of mixed MSW. Continuous feed, moving grate.
    FLUIDIZED_BED: Fluidized bed combustor using sand or similar
        media. Higher N2O due to lower combustion temperature.
    ROTARY_KILN: Rotary kiln incinerator for hazardous and
        industrial waste. Versatile feed acceptance.
    SEMI_CONTINUOUS: Semi-continuous feed incinerator with
        intermittent waste charging. Higher CH4 during startup.
    BATCH_TYPE: Batch-fed incinerator with discrete charging
        cycles. Highest CH4 from incomplete combustion.
    MODULAR: Modular starved-air incinerator with primary
        and secondary combustion chambers.
    """

    STOKER_GRATE = "stoker_grate"
    FLUIDIZED_BED = "fluidized_bed"
    ROTARY_KILN = "rotary_kiln"
    SEMI_CONTINUOUS = "semi_continuous"
    BATCH_TYPE = "batch_type"
    MODULAR = "modular"


class WastewaterSystem(str, Enum):
    """Wastewater treatment system types for MCF determination.

    The treatment system type determines the methane correction
    factor (MCF) for CH4 emission calculations from wastewater
    per IPCC 2006 Guidelines Volume 5 Chapter 6.

    AEROBIC_WELL_MANAGED: Well-designed and operated aerobic
        treatment (activated sludge, trickling filter). MCF = 0.0.
    AEROBIC_OVERLOADED: Aerobic system that is overloaded beyond
        design capacity, creating anaerobic zones. MCF = 0.3.
    ANAEROBIC_REACTOR: Enclosed anaerobic reactor (UASB, CSTR)
        without methane recovery. MCF = 0.8.
    ANAEROBIC_REACTOR_WITH_RECOVERY: Enclosed anaerobic reactor
        with biogas collection for flaring or energy. MCF = 0.8
        but CH4 recovered offsets emissions.
    ANAEROBIC_SHALLOW_LAGOON: Open anaerobic lagoon less than
        2 metres depth. Partial aerobic surface. MCF = 0.2.
    ANAEROBIC_DEEP_LAGOON: Open anaerobic lagoon greater than
        2 metres depth. Fully anaerobic below surface. MCF = 0.8.
    SEPTIC_SYSTEM: On-site septic tank and soil absorption system
        for decentralized wastewater treatment. MCF = 0.5.
    UNTREATED_DISCHARGE: Direct discharge of untreated wastewater
        to rivers, lakes, or ocean. MCF = 0.1.
    """

    AEROBIC_WELL_MANAGED = "aerobic_well_managed"
    AEROBIC_OVERLOADED = "aerobic_overloaded"
    ANAEROBIC_REACTOR = "anaerobic_reactor"
    ANAEROBIC_REACTOR_WITH_RECOVERY = "anaerobic_reactor_with_recovery"
    ANAEROBIC_SHALLOW_LAGOON = "anaerobic_shallow_lagoon"
    ANAEROBIC_DEEP_LAGOON = "anaerobic_deep_lagoon"
    SEPTIC_SYSTEM = "septic_system"
    UNTREATED_DISCHARGE = "untreated_discharge"


class CalculationMethod(str, Enum):
    """Methodology for calculating waste treatment emissions.

    Seven calculation methods spanning IPCC Tier 1 through Tier 3
    and supplementary approaches. The method determines the required
    input data granularity and expected uncertainty range.

    IPCC_FOD: IPCC First-Order Decay model for landfill methane
        generation over time. Uses decay rate constants and DOC.
        Formula: CH4(T) = DDOCm_decomp(T) * F * 16/12
    IPCC_TIER_1: IPCC Tier 1 using global default emission factors.
        Simplest approach with highest uncertainty. Suitable when
        only waste mass by category is available.
    IPCC_TIER_2: IPCC Tier 2 using country-specific or regional
        emission factors and waste composition data.
    IPCC_TIER_3: IPCC Tier 3 using facility-specific data,
        continuous emissions monitoring (CEMS), or detailed process
        models. Lowest uncertainty.
    MASS_BALANCE: Carbon mass balance approach tracking carbon
        inputs, outputs, and storage through the treatment process.
        Formula: Emissions = (C_in - C_out - C_stored) * 44/12
    DIRECT_MEASUREMENT: Direct emissions measurement using CEMS
        stack monitoring, ambient monitoring, or portable analyzers.
    SPEND_BASED: Using spend data with sector-specific emission
        factors from DEFRA, EPA, or Ecoinvent databases.
    """

    IPCC_FOD = "ipcc_fod"
    IPCC_TIER_1 = "ipcc_tier_1"
    IPCC_TIER_2 = "ipcc_tier_2"
    IPCC_TIER_3 = "ipcc_tier_3"
    MASS_BALANCE = "mass_balance"
    DIRECT_MEASUREMENT = "direct_measurement"
    SPEND_BASED = "spend_based"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in waste treatment emission calculations.

    CO2: Carbon dioxide - primary gas from fossil carbon in
        incineration, chemical treatment, and open burning.
        Biogenic CO2 tracked separately for reporting.
    CH4: Methane - from anaerobic decomposition in landfills,
        incomplete combustion, biogas leaks, and wastewater.
    N2O: Nitrous oxide - from nitrification/denitrification in
        wastewater, thermal NOx in combustion, and composting.
    CO: Carbon monoxide - from incomplete combustion in open
        burning and poorly controlled incineration. Precursor gas.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO = "CO"


class GWPSource(str, Enum):
    """IPCC Assessment Report source for Global Warming Potential values.

    Waste sector uniquely distinguishes fossil and biogenic CH4
    due to IPCC AR6 differentiating their GWP-100 values
    (29.8 for fossil CH4 vs 27.0 for biogenic CH4).

    AR4: Fourth Assessment Report (2007) - 100-year GWP.
    AR5: Fifth Assessment Report (2014) - 100-year GWP.
    AR6: Sixth Assessment Report (2021) - 100-year GWP.
    AR6_20YR: Sixth Assessment Report - 20-year GWP metric.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class EmissionFactorSource(str, Enum):
    """Authoritative source for waste treatment emission factors.

    IPCC_2006: IPCC 2006 Guidelines for National Greenhouse Gas
        Inventories, Volume 5 (Waste).
    IPCC_2019: 2019 Refinement to the 2006 IPCC Guidelines,
        Chapter 5 (updated biological treatment factors).
    EPA_AP42: US EPA AP-42 Compilation of Air Pollutant Emission
        Factors for waste treatment processes.
    DEFRA: UK DEFRA/BEIS Greenhouse Gas Conversion Factors
        (updated annually for waste sector).
    ECOINVENT: Ecoinvent database life cycle emission factors
        for waste treatment pathways.
    NATIONAL: Country-specific national inventory emission factors
        (Tier 2 data from NIR submissions).
    CUSTOM: User-provided custom emission factors from facility
        measurements or site-specific studies (Tier 3).
    """

    IPCC_2006 = "IPCC_2006"
    IPCC_2019 = "IPCC_2019"
    EPA_AP42 = "EPA_AP42"
    DEFRA = "DEFRA"
    ECOINVENT = "ECOINVENT"
    NATIONAL = "NATIONAL"
    CUSTOM = "CUSTOM"


class DataQualityTier(str, Enum):
    """IPCC data quality tier for input data and emission factors.

    TIER_1: Global default values from IPCC Guidelines. Highest
        uncertainty, lowest data requirements. Typical uncertainty
        range: +/-50% to +/-100%.
    TIER_2: Country-specific or regional data. Moderate uncertainty,
        requires national statistics or regional surveys. Typical
        uncertainty range: +/-25% to +/-50%.
    TIER_3: Facility-specific measured data. Lowest uncertainty,
        requires CEMS or detailed process measurements. Typical
        uncertainty range: +/-5% to +/-25%.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class FacilityType(str, Enum):
    """Type of waste treatment facility.

    Determines applicable regulatory requirements, emission factor
    sets, and reporting obligations.

    INDUSTRIAL_ONSITE: On-site industrial waste treatment at
        manufacturing or production facilities (Scope 1).
    MUNICIPAL_TREATMENT: Municipal solid waste treatment facility
        operated by or for local authorities.
    WASTE_TO_ENERGY: Dedicated waste-to-energy plant with
        electricity and/or heat generation.
    COMPOSTING_FACILITY: Dedicated composting facility for organic
        waste processing (windrow, in-vessel, or ASP).
    AD_PLANT: Dedicated anaerobic digestion plant for biogas
        production from organic waste.
    MBT_PLANT: Mechanical-biological treatment plant combining
        mechanical sorting with biological stabilization.
    WASTEWATER_PLANT: On-site wastewater treatment plant for
        industrial or process effluent.
    CHEMICAL_TREATMENT: Chemical treatment facility for hazardous
        waste neutralization, oxidation, or reduction.
    MULTI_STREAM: Multi-stream treatment facility handling multiple
        waste types and treatment methods.
    """

    INDUSTRIAL_ONSITE = "industrial_onsite"
    MUNICIPAL_TREATMENT = "municipal_treatment"
    WASTE_TO_ENERGY = "waste_to_energy"
    COMPOSTING_FACILITY = "composting_facility"
    AD_PLANT = "ad_plant"
    MBT_PLANT = "mbt_plant"
    WASTEWATER_PLANT = "wastewater_plant"
    CHEMICAL_TREATMENT = "chemical_treatment"
    MULTI_STREAM = "multi_stream"


class BiogasComponent(str, Enum):
    """Components of biogas from anaerobic digestion.

    Biogas composition affects energy content and emission
    calculations for anaerobic digestion facilities.

    METHANE: CH4 - primary energy-carrying component, typically
        50-70% of biogas volume.
    CARBON_DIOXIDE: CO2 - typically 30-50% of biogas volume,
        biogenic origin.
    HYDROGEN_SULFIDE: H2S - corrosive trace gas, typically
        0.01-0.5% requiring scrubbing.
    WATER_VAPOR: H2O - saturated at digester temperature,
        requires condensation for engine use.
    NITROGEN: N2 - trace component, typically <5%, from
        air ingress or feedstock nitrogen.
    OXYGEN: O2 - trace component, typically <1%, from
        air ingress at seals or feedstock.
    TRACE_GASES: Siloxanes, ammonia, and other trace components
        requiring monitoring for engine protection.
    """

    METHANE = "methane"
    CARBON_DIOXIDE = "carbon_dioxide"
    HYDROGEN_SULFIDE = "hydrogen_sulfide"
    WATER_VAPOR = "water_vapor"
    NITROGEN = "nitrogen"
    OXYGEN = "oxygen"
    TRACE_GASES = "trace_gases"


class ClimateZone(str, Enum):
    """Simplified IPCC climate zones for waste decomposition rates.

    Five climate zones used to stratify waste decomposition half-life
    values and landfill gas generation rates in the IPCC FOD model.
    Temperature and moisture determine microbial activity rates
    in landfill and biological treatment systems.

    TROPICAL: Tropical wet and dry climates. High decomposition
        rates, short half-lives. Mean annual temp > 20C.
    SUBTROPICAL: Subtropical climates. Moderate-to-high decomposition
        rates. Mean annual temp 15-20C.
    TEMPERATE: Temperate climates. Moderate decomposition rates.
        Mean annual temp 5-15C.
    BOREAL: Boreal and subarctic climates. Low decomposition rates,
        long half-lives. Mean annual temp 0-5C.
    POLAR: Polar and alpine climates. Very low decomposition rates.
        Mean annual temp < 0C.
    """

    TROPICAL = "tropical"
    SUBTROPICAL = "subtropical"
    TEMPERATE = "temperate"
    BOREAL = "boreal"
    POLAR = "polar"


class ComplianceStatus(str, Enum):
    """Result of a regulatory compliance check.

    COMPLIANT: All requirements of the regulatory framework are met.
    NON_COMPLIANT: One or more mandatory requirements are not met.
    PARTIAL: Some requirements are met but others are missing or
        incomplete.
    NOT_ASSESSED: Compliance has not been evaluated for this framework.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_ASSESSED = "not_assessed"


class ReportingPeriod(str, Enum):
    """Time granularity for emission aggregation and reporting.

    ANNUAL: Calendar year aggregation (most common for GHG reporting).
    QUARTERLY: Calendar quarter aggregation.
    MONTHLY: Calendar month aggregation.
    AD_HOC: User-defined custom date range aggregation.
    """

    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    AD_HOC = "ad_hoc"


class EmissionScope(str, Enum):
    """GHG Protocol emission scope classification.

    SCOPE_1: Direct emissions from owned or controlled on-site
        waste treatment operations.
    SCOPE_2: Indirect emissions from purchased electricity or
        heat used in waste treatment operations.
    SCOPE_3: Other indirect emissions, specifically GHG Protocol
        Scope 3 Category 5 (waste generated in operations sent
        to third-party treatment facilities).
    """

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


# =============================================================================
# Constant Tables (all Decimal for deterministic arithmetic)
# =============================================================================


# ---------------------------------------------------------------------------
# GWP values by IPCC Assessment Report
# Includes separate fossil and biogenic CH4 per IPCC AR6
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[GWPSource, Dict[str, Decimal]] = {
    GWPSource.AR4: {
        "CO2": Decimal("1"),
        "CH4": Decimal("25"),
        "CH4_biogenic": Decimal("25"),
        "N2O": Decimal("298"),
        "CO": Decimal("1.9"),
    },
    GWPSource.AR5: {
        "CO2": Decimal("1"),
        "CH4": Decimal("28"),
        "CH4_biogenic": Decimal("28"),
        "N2O": Decimal("265"),
        "CO": Decimal("1.9"),
    },
    GWPSource.AR6: {
        "CO2": Decimal("1"),
        "CH4": Decimal("29.8"),
        "CH4_biogenic": Decimal("27.0"),
        "N2O": Decimal("273"),
        "CO": Decimal("4.06"),
    },
    GWPSource.AR6_20YR: {
        "CO2": Decimal("1"),
        "CH4": Decimal("82.5"),
        "CH4_biogenic": Decimal("80.8"),
        "N2O": Decimal("273"),
        "CO": Decimal("4.06"),
    },
}


# ---------------------------------------------------------------------------
# Molecular weight conversion factors
# ---------------------------------------------------------------------------

#: CO2 to C molecular weight ratio (44/12).
CONVERSION_FACTOR_CO2_C: Decimal = Decimal("3.66667")

#: CH4 to C molecular weight ratio (16/12).
CH4_C_RATIO: Decimal = Decimal("1.33333")

#: N2O to N molecular weight ratio (44/28).
N2O_N_RATIO: Decimal = Decimal("1.57143")

#: CH4 density at Standard Temperature and Pressure (tonnes/m3).
CH4_DENSITY_STP: Decimal = Decimal("0.0007168")


# ---------------------------------------------------------------------------
# IPCC DOC values (Degradable Organic Carbon fraction)
# IPCC 2006 Guidelines Volume 5 Table 2.4 and 2019 Refinement
# Values represent fraction of wet waste weight that is DOC
# ---------------------------------------------------------------------------

IPCC_DOC_VALUES: Dict[WasteCategory, Decimal] = {
    WasteCategory.MSW: Decimal("0.15"),
    WasteCategory.INDUSTRIAL: Decimal("0.15"),
    WasteCategory.CONSTRUCTION_DEMOLITION: Decimal("0.08"),
    WasteCategory.ORGANIC: Decimal("0.15"),
    WasteCategory.FOOD: Decimal("0.15"),
    WasteCategory.YARD: Decimal("0.20"),
    WasteCategory.PAPER: Decimal("0.40"),
    WasteCategory.CARDBOARD: Decimal("0.40"),
    WasteCategory.PLASTIC: Decimal("0.0"),
    WasteCategory.METAL: Decimal("0.0"),
    WasteCategory.GLASS: Decimal("0.0"),
    WasteCategory.TEXTILES: Decimal("0.24"),
    WasteCategory.WOOD: Decimal("0.43"),
    WasteCategory.RUBBER: Decimal("0.0"),
    WasteCategory.E_WASTE: Decimal("0.0"),
    WasteCategory.HAZARDOUS: Decimal("0.0"),
    WasteCategory.MEDICAL: Decimal("0.05"),
    WasteCategory.SLUDGE: Decimal("0.05"),
    WasteCategory.MIXED: Decimal("0.12"),
}


# ---------------------------------------------------------------------------
# IPCC MCF values (Methane Correction Factor) by landfill/disposal type
# IPCC 2006 Guidelines Volume 5 Table 3.1
# MCF reflects the fraction of waste that decomposes anaerobically
# ---------------------------------------------------------------------------

IPCC_MCF_VALUES: Dict[str, Decimal] = {
    "managed_anaerobic": Decimal("1.0"),
    "managed_semi_aerobic": Decimal("0.5"),
    "unmanaged_deep": Decimal("0.8"),
    "unmanaged_shallow": Decimal("0.4"),
    "uncategorized": Decimal("0.6"),
    "open_dumping": Decimal("0.4"),
}


# ---------------------------------------------------------------------------
# IPCC Carbon Content by waste type
# IPCC 2006 Guidelines Volume 5 Table 5.2 and 2019 Refinement
# carbon_content_wet: fraction of total carbon in wet waste weight
# fossil_carbon_fraction: fraction of total carbon that is fossil origin
# dry_matter_fraction: fraction of wet waste that is dry matter
# ---------------------------------------------------------------------------

IPCC_CARBON_CONTENT: Dict[WasteCategory, Dict[str, Decimal]] = {
    WasteCategory.MSW: {
        "carbon_content_wet": Decimal("0.20"),
        "fossil_carbon_fraction": Decimal("0.40"),
        "dry_matter_fraction": Decimal("0.60"),
    },
    WasteCategory.INDUSTRIAL: {
        "carbon_content_wet": Decimal("0.25"),
        "fossil_carbon_fraction": Decimal("0.50"),
        "dry_matter_fraction": Decimal("0.70"),
    },
    WasteCategory.CONSTRUCTION_DEMOLITION: {
        "carbon_content_wet": Decimal("0.10"),
        "fossil_carbon_fraction": Decimal("0.10"),
        "dry_matter_fraction": Decimal("0.90"),
    },
    WasteCategory.ORGANIC: {
        "carbon_content_wet": Decimal("0.15"),
        "fossil_carbon_fraction": Decimal("0.0"),
        "dry_matter_fraction": Decimal("0.40"),
    },
    WasteCategory.FOOD: {
        "carbon_content_wet": Decimal("0.15"),
        "fossil_carbon_fraction": Decimal("0.0"),
        "dry_matter_fraction": Decimal("0.40"),
    },
    WasteCategory.YARD: {
        "carbon_content_wet": Decimal("0.18"),
        "fossil_carbon_fraction": Decimal("0.0"),
        "dry_matter_fraction": Decimal("0.40"),
    },
    WasteCategory.PAPER: {
        "carbon_content_wet": Decimal("0.36"),
        "fossil_carbon_fraction": Decimal("0.01"),
        "dry_matter_fraction": Decimal("0.90"),
    },
    WasteCategory.CARDBOARD: {
        "carbon_content_wet": Decimal("0.38"),
        "fossil_carbon_fraction": Decimal("0.01"),
        "dry_matter_fraction": Decimal("0.90"),
    },
    WasteCategory.PLASTIC: {
        "carbon_content_wet": Decimal("0.67"),
        "fossil_carbon_fraction": Decimal("1.0"),
        "dry_matter_fraction": Decimal("0.99"),
    },
    WasteCategory.METAL: {
        "carbon_content_wet": Decimal("0.0"),
        "fossil_carbon_fraction": Decimal("0.0"),
        "dry_matter_fraction": Decimal("0.99"),
    },
    WasteCategory.GLASS: {
        "carbon_content_wet": Decimal("0.0"),
        "fossil_carbon_fraction": Decimal("0.0"),
        "dry_matter_fraction": Decimal("0.99"),
    },
    WasteCategory.TEXTILES: {
        "carbon_content_wet": Decimal("0.45"),
        "fossil_carbon_fraction": Decimal("0.80"),
        "dry_matter_fraction": Decimal("0.80"),
    },
    WasteCategory.WOOD: {
        "carbon_content_wet": Decimal("0.46"),
        "fossil_carbon_fraction": Decimal("0.0"),
        "dry_matter_fraction": Decimal("0.85"),
    },
    WasteCategory.RUBBER: {
        "carbon_content_wet": Decimal("0.50"),
        "fossil_carbon_fraction": Decimal("0.20"),
        "dry_matter_fraction": Decimal("0.84"),
    },
    WasteCategory.E_WASTE: {
        "carbon_content_wet": Decimal("0.10"),
        "fossil_carbon_fraction": Decimal("0.80"),
        "dry_matter_fraction": Decimal("0.95"),
    },
    WasteCategory.HAZARDOUS: {
        "carbon_content_wet": Decimal("0.15"),
        "fossil_carbon_fraction": Decimal("0.60"),
        "dry_matter_fraction": Decimal("0.75"),
    },
    WasteCategory.MEDICAL: {
        "carbon_content_wet": Decimal("0.25"),
        "fossil_carbon_fraction": Decimal("0.50"),
        "dry_matter_fraction": Decimal("0.70"),
    },
    WasteCategory.SLUDGE: {
        "carbon_content_wet": Decimal("0.10"),
        "fossil_carbon_fraction": Decimal("0.0"),
        "dry_matter_fraction": Decimal("0.25"),
    },
    WasteCategory.MIXED: {
        "carbon_content_wet": Decimal("0.22"),
        "fossil_carbon_fraction": Decimal("0.35"),
        "dry_matter_fraction": Decimal("0.60"),
    },
}


# ---------------------------------------------------------------------------
# IPCC Composting / Biological Treatment Emission Factors
# IPCC 2019 Refinement Table 5.1
# CH4 and N2O in g per kg of waste treated (wet weight)
# ---------------------------------------------------------------------------

IPCC_COMPOSTING_EF: Dict[str, Dict[str, Decimal]] = {
    "composting_well_managed": {
        "CH4": Decimal("4.0"),
        "N2O": Decimal("0.24"),
    },
    "composting_poorly_managed": {
        "CH4": Decimal("10.0"),
        "N2O": Decimal("0.6"),
    },
    "anaerobic_digestion_vented": {
        "CH4": Decimal("2.0"),
        "N2O": Decimal("0.0"),
    },
    "anaerobic_digestion_flared": {
        "CH4": Decimal("0.8"),
        "N2O": Decimal("0.0"),
    },
    "anaerobic_digestion_utilized": {
        "CH4": Decimal("0.5"),
        "N2O": Decimal("0.0"),
    },
    "mbt_aerobic": {
        "CH4": Decimal("4.0"),
        "N2O": Decimal("0.3"),
    },
    "mbt_anaerobic_pretreatment": {
        "CH4": Decimal("2.0"),
        "N2O": Decimal("0.1"),
    },
    "vermicomposting": {
        "CH4": Decimal("2.5"),
        "N2O": Decimal("0.12"),
    },
    "home_composting": {
        "CH4": Decimal("8.0"),
        "N2O": Decimal("0.4"),
    },
}


# ---------------------------------------------------------------------------
# IPCC Incineration Emission Factors by incinerator technology
# IPCC 2006 Guidelines Volume 5 Table 5.3
# N2O in kg per Gg (1000 tonnes) of waste incinerated
# CH4 in kg per Gg (1000 tonnes) of waste incinerated
# ---------------------------------------------------------------------------

IPCC_INCINERATION_EF: Dict[IncineratorType, Dict[str, Decimal]] = {
    IncineratorType.STOKER_GRATE: {
        "N2O": Decimal("50"),
        "CH4": Decimal("0.2"),
    },
    IncineratorType.FLUIDIZED_BED: {
        "N2O": Decimal("56"),
        "CH4": Decimal("0.68"),
    },
    IncineratorType.ROTARY_KILN: {
        "N2O": Decimal("50"),
        "CH4": Decimal("0.2"),
    },
    IncineratorType.SEMI_CONTINUOUS: {
        "N2O": Decimal("60"),
        "CH4": Decimal("6.0"),
    },
    IncineratorType.BATCH_TYPE: {
        "N2O": Decimal("60"),
        "CH4": Decimal("60"),
    },
    IncineratorType.MODULAR: {
        "N2O": Decimal("55"),
        "CH4": Decimal("3.0"),
    },
}


# ---------------------------------------------------------------------------
# IPCC Wastewater MCF values by treatment system type
# IPCC 2006 Guidelines Volume 5 Chapter 6 Table 6.3
# ---------------------------------------------------------------------------

IPCC_WASTEWATER_MCF: Dict[WastewaterSystem, Decimal] = {
    WastewaterSystem.AEROBIC_WELL_MANAGED: Decimal("0.0"),
    WastewaterSystem.AEROBIC_OVERLOADED: Decimal("0.3"),
    WastewaterSystem.ANAEROBIC_REACTOR: Decimal("0.8"),
    WastewaterSystem.ANAEROBIC_REACTOR_WITH_RECOVERY: Decimal("0.8"),
    WastewaterSystem.ANAEROBIC_SHALLOW_LAGOON: Decimal("0.2"),
    WastewaterSystem.ANAEROBIC_DEEP_LAGOON: Decimal("0.8"),
    WastewaterSystem.SEPTIC_SYSTEM: Decimal("0.5"),
    WastewaterSystem.UNTREATED_DISCHARGE: Decimal("0.1"),
}


# ---------------------------------------------------------------------------
# IPCC Maximum CH4 producing capacity (Bo)
# kg CH4 per kg BOD for domestic wastewater = 0.6
# kg CH4 per kg COD for domestic wastewater = 0.25
# Industrial wastewater varies by industry type
# ---------------------------------------------------------------------------

WASTEWATER_BO: Dict[str, Decimal] = {
    "domestic_bod": Decimal("0.6"),
    "domestic_cod": Decimal("0.25"),
    "industrial_pulp_paper_cod": Decimal("0.25"),
    "industrial_food_processing_cod": Decimal("0.25"),
    "industrial_dairy_cod": Decimal("0.25"),
    "industrial_brewery_cod": Decimal("0.25"),
    "industrial_slaughterhouse_cod": Decimal("0.25"),
    "industrial_chemical_cod": Decimal("0.25"),
    "industrial_pharmaceutical_cod": Decimal("0.25"),
    "industrial_refinery_cod": Decimal("0.25"),
}


# ---------------------------------------------------------------------------
# Wastewater N2O emission factors
# IPCC 2006 Guidelines Volume 5 Chapter 6
# Values in kg N2O-N per kg N
# ---------------------------------------------------------------------------

WASTEWATER_N2O_EF: Dict[str, Decimal] = {
    "plant_ef": Decimal("0.016"),
    "effluent_ef": Decimal("0.005"),
    "nitrogen_fraction_protein": Decimal("0.16"),
}


# ---------------------------------------------------------------------------
# Net Calorific Value (NCV) by waste type in GJ per tonne (wet basis)
# Used for energy recovery credit calculations
# IPCC 2006 Guidelines Volume 5 and literature values
# ---------------------------------------------------------------------------

INCINERATION_NCV: Dict[WasteCategory, Decimal] = {
    WasteCategory.MSW: Decimal("9.0"),
    WasteCategory.INDUSTRIAL: Decimal("12.0"),
    WasteCategory.CONSTRUCTION_DEMOLITION: Decimal("5.0"),
    WasteCategory.ORGANIC: Decimal("4.0"),
    WasteCategory.FOOD: Decimal("3.5"),
    WasteCategory.YARD: Decimal("5.5"),
    WasteCategory.PAPER: Decimal("13.0"),
    WasteCategory.CARDBOARD: Decimal("14.0"),
    WasteCategory.PLASTIC: Decimal("32.0"),
    WasteCategory.METAL: Decimal("0.0"),
    WasteCategory.GLASS: Decimal("0.0"),
    WasteCategory.TEXTILES: Decimal("16.0"),
    WasteCategory.WOOD: Decimal("15.0"),
    WasteCategory.RUBBER: Decimal("26.0"),
    WasteCategory.E_WASTE: Decimal("5.0"),
    WasteCategory.HAZARDOUS: Decimal("10.0"),
    WasteCategory.MEDICAL: Decimal("14.0"),
    WasteCategory.SLUDGE: Decimal("2.0"),
    WasteCategory.MIXED: Decimal("8.5"),
}


# ---------------------------------------------------------------------------
# Half-life values for waste decomposition (years)
# Used in IPCC First-Order Decay (FOD) model
# k = ln(2) / half_life
# Keyed by (ClimateZone, waste_degradability_class)
# IPCC 2006 Guidelines Volume 5 Table 3.3 and 2019 Refinement
# ---------------------------------------------------------------------------

HALF_LIFE_VALUES: Dict[Tuple[ClimateZone, str], Decimal] = {
    # Tropical (wet/dry) - rapid decomposition
    (ClimateZone.TROPICAL, "rapidly_degrading"): Decimal("3"),
    (ClimateZone.TROPICAL, "moderately_degrading"): Decimal("5"),
    (ClimateZone.TROPICAL, "slowly_degrading"): Decimal("10"),
    (ClimateZone.TROPICAL, "very_slowly_degrading"): Decimal("20"),
    # Subtropical - moderate-rapid decomposition
    (ClimateZone.SUBTROPICAL, "rapidly_degrading"): Decimal("4"),
    (ClimateZone.SUBTROPICAL, "moderately_degrading"): Decimal("7"),
    (ClimateZone.SUBTROPICAL, "slowly_degrading"): Decimal("14"),
    (ClimateZone.SUBTROPICAL, "very_slowly_degrading"): Decimal("25"),
    # Temperate - moderate decomposition
    (ClimateZone.TEMPERATE, "rapidly_degrading"): Decimal("5"),
    (ClimateZone.TEMPERATE, "moderately_degrading"): Decimal("10"),
    (ClimateZone.TEMPERATE, "slowly_degrading"): Decimal("20"),
    (ClimateZone.TEMPERATE, "very_slowly_degrading"): Decimal("35"),
    # Boreal - slow decomposition
    (ClimateZone.BOREAL, "rapidly_degrading"): Decimal("7"),
    (ClimateZone.BOREAL, "moderately_degrading"): Decimal("14"),
    (ClimateZone.BOREAL, "slowly_degrading"): Decimal("28"),
    (ClimateZone.BOREAL, "very_slowly_degrading"): Decimal("50"),
    # Polar - very slow decomposition
    (ClimateZone.POLAR, "rapidly_degrading"): Decimal("10"),
    (ClimateZone.POLAR, "moderately_degrading"): Decimal("20"),
    (ClimateZone.POLAR, "slowly_degrading"): Decimal("40"),
    (ClimateZone.POLAR, "very_slowly_degrading"): Decimal("70"),
}


# ---------------------------------------------------------------------------
# Waste degradability classification by waste category
# Maps each waste type to its FOD degradability class
# ---------------------------------------------------------------------------

WASTE_DEGRADABILITY_CLASS: Dict[WasteCategory, str] = {
    WasteCategory.MSW: "moderately_degrading",
    WasteCategory.INDUSTRIAL: "moderately_degrading",
    WasteCategory.CONSTRUCTION_DEMOLITION: "slowly_degrading",
    WasteCategory.ORGANIC: "rapidly_degrading",
    WasteCategory.FOOD: "rapidly_degrading",
    WasteCategory.YARD: "moderately_degrading",
    WasteCategory.PAPER: "moderately_degrading",
    WasteCategory.CARDBOARD: "moderately_degrading",
    WasteCategory.PLASTIC: "very_slowly_degrading",
    WasteCategory.METAL: "very_slowly_degrading",
    WasteCategory.GLASS: "very_slowly_degrading",
    WasteCategory.TEXTILES: "slowly_degrading",
    WasteCategory.WOOD: "slowly_degrading",
    WasteCategory.RUBBER: "very_slowly_degrading",
    WasteCategory.E_WASTE: "very_slowly_degrading",
    WasteCategory.HAZARDOUS: "slowly_degrading",
    WasteCategory.MEDICAL: "moderately_degrading",
    WasteCategory.SLUDGE: "rapidly_degrading",
    WasteCategory.MIXED: "moderately_degrading",
}


# ---------------------------------------------------------------------------
# Open burning emission factors
# g gas per kg dry matter burned
# IPCC 2006 Guidelines Volume 5 Section 5.3.2
# ---------------------------------------------------------------------------

OPEN_BURNING_EF: Dict[EmissionGas, Decimal] = {
    EmissionGas.CO2: Decimal("1550"),
    EmissionGas.CH4: Decimal("6.5"),
    EmissionGas.N2O: Decimal("0.15"),
    EmissionGas.CO: Decimal("69"),
}


# ---------------------------------------------------------------------------
# Biochemical Methane Potential (BMP) default values
# m3 CH4 per tonne of volatile solids (VS) at STP
# Used for anaerobic digestion biogas yield estimation
# ---------------------------------------------------------------------------

BMP_DEFAULTS: Dict[WasteCategory, Decimal] = {
    WasteCategory.FOOD: Decimal("400"),
    WasteCategory.YARD: Decimal("250"),
    WasteCategory.ORGANIC: Decimal("350"),
    WasteCategory.PAPER: Decimal("200"),
    WasteCategory.CARDBOARD: Decimal("210"),
    WasteCategory.SLUDGE: Decimal("280"),
    WasteCategory.MSW: Decimal("180"),
    WasteCategory.MIXED: Decimal("160"),
}


# ---------------------------------------------------------------------------
# Volatile Solids (VS) fraction of dry matter by waste type
# Used for anaerobic digestion calculations
# ---------------------------------------------------------------------------

VS_FRACTION: Dict[WasteCategory, Decimal] = {
    WasteCategory.FOOD: Decimal("0.87"),
    WasteCategory.YARD: Decimal("0.70"),
    WasteCategory.ORGANIC: Decimal("0.80"),
    WasteCategory.PAPER: Decimal("0.82"),
    WasteCategory.CARDBOARD: Decimal("0.80"),
    WasteCategory.SLUDGE: Decimal("0.65"),
    WasteCategory.MSW: Decimal("0.55"),
    WasteCategory.MIXED: Decimal("0.50"),
}


# ---------------------------------------------------------------------------
# Default methane fraction in biogas by feedstock type
# ---------------------------------------------------------------------------

BIOGAS_CH4_FRACTION: Dict[WasteCategory, Decimal] = {
    WasteCategory.FOOD: Decimal("0.60"),
    WasteCategory.YARD: Decimal("0.55"),
    WasteCategory.ORGANIC: Decimal("0.58"),
    WasteCategory.PAPER: Decimal("0.52"),
    WasteCategory.CARDBOARD: Decimal("0.52"),
    WasteCategory.SLUDGE: Decimal("0.65"),
    WasteCategory.MSW: Decimal("0.55"),
    WasteCategory.MIXED: Decimal("0.53"),
}


# ---------------------------------------------------------------------------
# Pyrolysis and gasification emission factors
# kg CO2e per tonne of waste treated
# Literature-derived values for advanced thermal treatment
# ---------------------------------------------------------------------------

ADVANCED_THERMAL_EF: Dict[str, Dict[str, Decimal]] = {
    "pyrolysis": {
        "CO2_fossil": Decimal("500"),
        "CH4": Decimal("2.0"),
        "N2O": Decimal("0.5"),
        "CO": Decimal("15.0"),
    },
    "gasification": {
        "CO2_fossil": Decimal("600"),
        "CH4": Decimal("1.5"),
        "N2O": Decimal("0.3"),
        "CO": Decimal("20.0"),
    },
}


# =============================================================================
# Pydantic Data Models (18)
# =============================================================================


class WasteComposition(BaseModel):
    """Waste stream composition breakdown by waste category.

    Represents the fractional composition of a waste stream,
    where all fractions must sum to 1.0 (100%). Used to calculate
    weighted-average emission factors and carbon content for
    mixed waste streams.

    Attributes:
        id: Unique composition record identifier (UUID).
        name: Human-readable name for the composition profile.
        fractions: Dictionary mapping WasteCategory value to mass
            fraction (Decimal, 0.0-1.0).
        moisture_content: Overall moisture content as fraction
            (0.0-1.0), used for NCV and dry matter adjustments.
        total_carbon_content: Weighted average total carbon content
            as fraction of wet weight.
        fossil_carbon_fraction: Weighted average fossil carbon
            fraction of total carbon.
        doc_weighted: Weighted average DOC fraction.
        notes: Optional description or source of composition data.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique composition record identifier (UUID)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable composition profile name",
    )
    fractions: Dict[str, Decimal] = Field(
        ...,
        description=(
            "Mapping of WasteCategory value to mass fraction (0.0-1.0)"
        ),
    )
    moisture_content: Decimal = Field(
        default=Decimal("0.30"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Overall moisture content as fraction",
    )
    total_carbon_content: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Weighted average total carbon content fraction",
    )
    fossil_carbon_fraction: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Weighted average fossil carbon fraction",
    )
    doc_weighted: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Weighted average DOC fraction",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional description or source of composition data",
    )

    @field_validator("fractions")
    @classmethod
    def validate_fractions(
        cls, v: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """Validate that fractions are non-negative and keys are valid."""
        if not v:
            raise ValueError("At least one waste fraction must be specified")
        for key, fraction in v.items():
            if fraction < Decimal("0"):
                raise ValueError(
                    f"Fraction for '{key}' must be non-negative, "
                    f"got {fraction}"
                )
            if fraction > Decimal("1.0"):
                raise ValueError(
                    f"Fraction for '{key}' must be <= 1.0, got {fraction}"
                )
        total = sum(v.values())
        if total > Decimal("0") and abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Fractions must sum to 1.0 (within 1% tolerance), "
                f"got {total}"
            )
        return v


class TreatmentFacilityInfo(BaseModel):
    """Treatment facility registration and metadata.

    Represents a waste treatment facility with its operational
    characteristics, location, permitted methods, and capacity.
    Each facility is registered under a tenant for multi-tenancy.

    Attributes:
        id: Unique facility identifier (UUID).
        name: Human-readable facility name.
        facility_type: Type of treatment facility.
        treatment_methods: List of treatment methods operated.
        capacity_tonnes_per_year: Annual treatment capacity in tonnes.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        climate_zone: Climate zone for decomposition rate selection.
        country_code: ISO 3166-1 alpha-2 country code.
        permit_number: Environmental permit or license number.
        tenant_id: Owning tenant identifier for multi-tenancy.
        is_active: Whether facility is currently operational.
        commission_date: Date facility was commissioned.
        created_at: UTC timestamp of facility registration.
        updated_at: UTC timestamp of last update.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique facility identifier (UUID)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable facility name",
    )
    facility_type: FacilityType = Field(
        ...,
        description="Type of treatment facility",
    )
    treatment_methods: List[TreatmentMethod] = Field(
        ...,
        min_length=1,
        description="List of treatment methods operated at facility",
    )
    capacity_tonnes_per_year: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Annual treatment capacity in tonnes",
    )
    latitude: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="WGS84 latitude in decimal degrees",
    )
    longitude: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="WGS84 longitude in decimal degrees",
    )
    climate_zone: ClimateZone = Field(
        default=ClimateZone.TEMPERATE,
        description="Climate zone for decomposition rate selection",
    )
    country_code: str = Field(
        default="",
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    permit_number: str = Field(
        default="",
        max_length=200,
        description="Environmental permit or license number",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Owning tenant identifier",
    )
    is_active: bool = Field(
        default=True,
        description="Whether facility is currently operational",
    )
    commission_date: Optional[datetime] = Field(
        default=None,
        description="Date facility was commissioned",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of facility registration",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of last update",
    )

    @field_validator("treatment_methods")
    @classmethod
    def validate_treatment_methods(
        cls, v: List[TreatmentMethod]
    ) -> List[TreatmentMethod]:
        """Validate treatment methods list has no duplicates."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate treatment methods are not allowed")
        return v


class WasteStreamInfo(BaseModel):
    """Waste stream definition with composition and volume.

    Represents a defined waste stream entering a treatment facility,
    including its source, composition profile, volume, and treatment
    assignment.

    Attributes:
        id: Unique waste stream identifier (UUID).
        name: Human-readable waste stream name.
        facility_id: Reference to the treatment facility.
        waste_category: Primary waste category classification.
        composition: Optional detailed composition breakdown.
        annual_volume_tonnes: Annual waste volume in tonnes.
        treatment_method: Assigned treatment method.
        source_description: Description of waste source/origin.
        scope: GHG Protocol emission scope for this stream.
        data_quality_tier: Quality tier of the waste data.
        tenant_id: Owning tenant identifier.
        created_at: UTC timestamp of stream registration.
        updated_at: UTC timestamp of last update.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique waste stream identifier (UUID)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable waste stream name",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the treatment facility",
    )
    waste_category: WasteCategory = Field(
        ...,
        description="Primary waste category classification",
    )
    composition: Optional[WasteComposition] = Field(
        default=None,
        description="Optional detailed composition breakdown",
    )
    annual_volume_tonnes: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Annual waste volume in tonnes",
    )
    treatment_method: TreatmentMethod = Field(
        ...,
        description="Assigned treatment method",
    )
    source_description: str = Field(
        default="",
        max_length=2000,
        description="Description of waste source or origin",
    )
    scope: EmissionScope = Field(
        default=EmissionScope.SCOPE_1,
        description="GHG Protocol emission scope for this stream",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Quality tier of the waste data",
    )
    tenant_id: str = Field(
        default="",
        description="Owning tenant identifier",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of stream registration",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of last update",
    )


class EmissionFactorRecord(BaseModel):
    """Emission factor record for waste treatment calculations.

    Stores a single emission factor with its source, applicability
    scope, and units for use in waste treatment emission calculations.

    Attributes:
        id: Unique emission factor record identifier (UUID).
        waste_category: Waste category the factor applies to.
        treatment_method: Treatment method the factor applies to.
        gas: Greenhouse gas species.
        ef_value: Emission factor value.
        ef_unit: Unit of the emission factor (e.g. kg/tonne, g/kg).
        source: Authoritative source of the factor.
        data_quality_tier: IPCC tier level of the factor.
        climate_zone: Optional climate zone scoping.
        incinerator_type: Optional incinerator type scoping.
        valid_from: Start date of factor validity period.
        valid_to: End date of factor validity period.
        reference: Bibliographic reference string.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique emission factor record identifier (UUID)",
    )
    waste_category: Optional[WasteCategory] = Field(
        default=None,
        description="Waste category the factor applies to",
    )
    treatment_method: Optional[TreatmentMethod] = Field(
        default=None,
        description="Treatment method the factor applies to",
    )
    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species",
    )
    ef_value: Decimal = Field(
        ...,
        description="Emission factor value",
    )
    ef_unit: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unit of the emission factor",
    )
    source: EmissionFactorSource = Field(
        default=EmissionFactorSource.IPCC_2006,
        description="Authoritative source of the factor",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="IPCC tier level of the factor",
    )
    climate_zone: Optional[ClimateZone] = Field(
        default=None,
        description="Optional climate zone scoping",
    )
    incinerator_type: Optional[IncineratorType] = Field(
        default=None,
        description="Optional incinerator type scoping",
    )
    valid_from: Optional[datetime] = Field(
        default=None,
        description="Start date of factor validity period",
    )
    valid_to: Optional[datetime] = Field(
        default=None,
        description="End date of factor validity period",
    )
    reference: str = Field(
        default="",
        max_length=1000,
        description="Bibliographic reference string",
    )


class BiologicalTreatmentInput(BaseModel):
    """Input parameters specific to biological treatment methods.

    Captures the additional parameters needed for composting,
    anaerobic digestion, and MBT emission calculations beyond
    the base CalculationRequest fields.

    Attributes:
        composting_type: Type of composting system.
        is_well_managed: Whether the system is well-managed
            (affects default EF selection).
        ch4_recovery_fraction: Fraction of CH4 recovered by
            biofilter or gas collection (0.0-1.0).
        volatile_solids_fraction: Fraction of dry matter that
            is volatile solids (for AD calculations).
        bmp: Biochemical methane potential (m3 CH4/tonne VS).
        digestion_efficiency: AD digestion efficiency (0.0-1.0).
        ch4_fraction_biogas: Methane fraction in biogas (0.0-1.0).
        biogas_capture_efficiency: Fraction of biogas captured
            (0.0-1.0).
        biogas_flare_fraction: Fraction of captured biogas sent
            to flare (0.0-1.0).
        biogas_utilization_fraction: Fraction of captured biogas
            utilized for energy (0.0-1.0).
        biogas_vent_fraction: Fraction of captured biogas vented
            (0.0-1.0).
        flare_destruction_efficiency: CH4 destruction efficiency
            of flare (0.0-1.0).
        residence_time_days: Composting/digestion residence time.
        temperature_celsius: Operating temperature.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    composting_type: Optional[CompostingType] = Field(
        default=None,
        description="Type of composting system",
    )
    is_well_managed: bool = Field(
        default=True,
        description="Whether the treatment system is well-managed",
    )
    ch4_recovery_fraction: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of CH4 recovered by biofilter",
    )
    volatile_solids_fraction: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of dry matter that is volatile solids",
    )
    bmp: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Biochemical methane potential (m3 CH4/tonne VS)",
    )
    digestion_efficiency: Decimal = Field(
        default=Decimal("0.7"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Anaerobic digestion efficiency",
    )
    ch4_fraction_biogas: Decimal = Field(
        default=Decimal("0.60"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Methane fraction in biogas",
    )
    biogas_capture_efficiency: Decimal = Field(
        default=Decimal("0.95"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of biogas captured",
    )
    biogas_flare_fraction: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of captured biogas sent to flare",
    )
    biogas_utilization_fraction: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of captured biogas utilized for energy",
    )
    biogas_vent_fraction: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of captured biogas vented",
    )
    flare_destruction_efficiency: Decimal = Field(
        default=Decimal("0.98"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="CH4 destruction efficiency of flare",
    )
    residence_time_days: Optional[int] = Field(
        default=None,
        gt=0,
        le=365,
        description="Treatment residence time in days",
    )
    temperature_celsius: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("-10"),
        le=Decimal("80"),
        description="Operating temperature in degrees Celsius",
    )

    @field_validator("biogas_flare_fraction")
    @classmethod
    def validate_biogas_routing(cls, v: Decimal, info: Any) -> Decimal:
        """Validate that biogas routing fractions do not exceed 1.0."""
        data = info.data if hasattr(info, "data") else {}
        flare = v
        utilize = data.get("biogas_utilization_fraction", Decimal("0"))
        vent = data.get("biogas_vent_fraction", Decimal("0"))
        if utilize is not None and vent is not None:
            total = flare + utilize + vent
            if total > Decimal("1.01"):
                raise ValueError(
                    f"Biogas routing fractions (flare={flare}, "
                    f"utilize={utilize}, vent={vent}) must sum to <= 1.0, "
                    f"got {total}"
                )
        return v


class ThermalTreatmentInput(BaseModel):
    """Input parameters specific to thermal treatment methods.

    Captures the additional parameters needed for incineration,
    pyrolysis, gasification, and open burning calculations
    including fossil/biogenic carbon separation and energy recovery.

    Attributes:
        incinerator_type: Type of incinerator technology.
        oxidation_factor: Fraction of carbon oxidized during
            combustion (0.0-1.0, default 1.0 for modern).
        burnout_efficiency: Fraction of waste mass combusted
            (0.0-1.0, default 0.97).
        fossil_carbon_override: Optional override for fossil
            carbon fraction (overrides IPCC default).
        include_biogenic_co2: Whether to include biogenic CO2
            in reported emissions (normally excluded).
        energy_recovery_efficiency: Thermal energy recovery
            efficiency for WtE (0.0-1.0).
        electricity_generation_efficiency: Electrical conversion
            efficiency for WtE (0.0-1.0).
        heat_generation_efficiency: Heat recovery efficiency
            for combined heat and power (0.0-1.0).
        grid_emission_factor: Grid emission factor for energy
            offset calculation (tCO2e/GJ).
        flue_gas_treatment: Type of flue gas treatment system.
        continuous_monitoring: Whether CEMS is installed.
        air_pollution_control: Description of APC equipment.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    incinerator_type: Optional[IncineratorType] = Field(
        default=None,
        description="Type of incinerator technology",
    )
    oxidation_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of carbon oxidized during combustion",
    )
    burnout_efficiency: Decimal = Field(
        default=Decimal("0.97"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of waste mass combusted",
    )
    fossil_carbon_override: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Override for fossil carbon fraction",
    )
    include_biogenic_co2: bool = Field(
        default=False,
        description="Whether to include biogenic CO2 in emissions",
    )
    energy_recovery_efficiency: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Thermal energy recovery efficiency for WtE",
    )
    electricity_generation_efficiency: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("0.5"),
        description="Electrical conversion efficiency",
    )
    heat_generation_efficiency: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("0.9"),
        description="Heat recovery efficiency for CHP",
    )
    grid_emission_factor: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        description="Grid emission factor (tCO2e/GJ) for offset",
    )
    flue_gas_treatment: str = Field(
        default="",
        max_length=500,
        description="Type of flue gas treatment system",
    )
    continuous_monitoring: bool = Field(
        default=False,
        description="Whether CEMS is installed",
    )
    air_pollution_control: str = Field(
        default="",
        max_length=500,
        description="Description of APC equipment",
    )


class WastewaterTreatmentInput(BaseModel):
    """Input parameters specific to wastewater treatment.

    Captures the additional parameters needed for on-site
    industrial and process wastewater CH4 and N2O calculations
    per IPCC 2006 Guidelines Volume 5 Chapter 6.

    Attributes:
        system_type: Type of wastewater treatment system.
        total_organic_waste_kg: Total organic waste in wastewater
            (kg BOD or COD per year).
        organic_basis: Whether TOW is measured as BOD or COD.
        sludge_removal_kg: Organic component removed as sludge
            (kg BOD or COD per year).
        ch4_recovered_tonnes: CH4 recovered from treatment
            (tonnes per year).
        population_equivalent: Population or production equivalent
            for N2O calculations.
        protein_consumption_kg_per_person: Annual per-capita
            protein consumption (kg/person/yr).
        nitrogen_fraction_protein: Fraction of nitrogen in protein.
        n2o_plant_ef: N2O emission factor for treatment plant
            (kg N2O-N per kg N).
        n2o_effluent_ef: N2O emission factor for discharged
            effluent (kg N2O-N per kg N).
        nitrogen_effluent_kg: Total nitrogen in effluent discharge
            (kg N per year).
        industrial_sector: Industry sector for wastewater type.
        discharge_destination: Where treated effluent is discharged.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    system_type: WastewaterSystem = Field(
        ...,
        description="Type of wastewater treatment system",
    )
    total_organic_waste_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total organic waste (kg BOD or COD per year)",
    )
    organic_basis: str = Field(
        default="cod",
        description="Measurement basis: 'bod' or 'cod'",
    )
    sludge_removal_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Organic component removed as sludge (kg/yr)",
    )
    ch4_recovered_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 recovered from treatment (tonnes/yr)",
    )
    population_equivalent: Optional[int] = Field(
        default=None,
        ge=0,
        description="Population equivalent for N2O calculations",
    )
    protein_consumption_kg_per_person: Decimal = Field(
        default=Decimal("25.0"),
        ge=Decimal("0"),
        description="Annual per-capita protein consumption (kg/person/yr)",
    )
    nitrogen_fraction_protein: Decimal = Field(
        default=Decimal("0.16"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of nitrogen in protein",
    )
    n2o_plant_ef: Decimal = Field(
        default=Decimal("0.016"),
        ge=Decimal("0"),
        description="N2O plant EF (kg N2O-N per kg N)",
    )
    n2o_effluent_ef: Decimal = Field(
        default=Decimal("0.005"),
        ge=Decimal("0"),
        description="N2O effluent EF (kg N2O-N per kg N)",
    )
    nitrogen_effluent_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total nitrogen in effluent discharge (kg N/yr)",
    )
    industrial_sector: str = Field(
        default="",
        max_length=200,
        description="Industry sector for wastewater type",
    )
    discharge_destination: str = Field(
        default="",
        max_length=200,
        description="Where treated effluent is discharged",
    )

    @field_validator("organic_basis")
    @classmethod
    def validate_organic_basis(cls, v: str) -> str:
        """Validate organic basis is either BOD or COD."""
        v_lower = v.lower().strip()
        if v_lower not in ("bod", "cod"):
            raise ValueError(
                f"organic_basis must be 'bod' or 'cod', got '{v}'"
            )
        return v_lower


class MethaneRecoveryRecord(BaseModel):
    """Record of methane recovery, flaring, and utilization.

    Tracks CH4 capture and routing for a treatment facility,
    used to calculate net CH4 emissions after recovery.

    Attributes:
        id: Unique record identifier (UUID).
        facility_id: Reference to the treatment facility.
        period_start: Start of the recording period.
        period_end: End of the recording period.
        ch4_generated_tonnes: Total CH4 generated (tonnes).
        ch4_captured_tonnes: Total CH4 captured (tonnes).
        collection_efficiency: Gas collection efficiency (0.0-1.0).
        ch4_flared_tonnes: CH4 sent to flare (tonnes).
        ch4_utilized_tonnes: CH4 utilized for energy (tonnes).
        ch4_vented_tonnes: CH4 vented to atmosphere (tonnes).
        flare_destruction_efficiency: Flare destruction eff (0.0-1.0).
        utilization_conversion_efficiency: Engine/turbine conversion
            efficiency for CH4 utilization (0.0-1.0).
        electricity_generated_mwh: Electricity generated from CH4.
        heat_generated_gj: Heat generated from CH4.
        notes: Optional notes about the recovery event.
        tenant_id: Owning tenant identifier.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID)",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the treatment facility",
    )
    period_start: datetime = Field(
        ...,
        description="Start of the recording period",
    )
    period_end: datetime = Field(
        ...,
        description="End of the recording period",
    )
    ch4_generated_tonnes: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CH4 generated (tonnes)",
    )
    ch4_captured_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total CH4 captured (tonnes)",
    )
    collection_efficiency: Decimal = Field(
        default=Decimal("0.75"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Gas collection efficiency",
    )
    ch4_flared_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 sent to flare (tonnes)",
    )
    ch4_utilized_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 utilized for energy (tonnes)",
    )
    ch4_vented_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="CH4 vented to atmosphere (tonnes)",
    )
    flare_destruction_efficiency: Decimal = Field(
        default=Decimal("0.98"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Flare destruction efficiency",
    )
    utilization_conversion_efficiency: Decimal = Field(
        default=Decimal("0.95"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Engine/turbine conversion efficiency",
    )
    electricity_generated_mwh: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Electricity generated from CH4 utilization (MWh)",
    )
    heat_generated_gj: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Heat generated from CH4 utilization (GJ)",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes about recovery event",
    )
    tenant_id: str = Field(
        default="",
        description="Owning tenant identifier",
    )

    @field_validator("period_end")
    @classmethod
    def validate_period_end(cls, v: datetime, info: Any) -> datetime:
        """Validate that period_end is after period_start."""
        data = info.data if hasattr(info, "data") else {}
        period_start = data.get("period_start")
        if period_start is not None and v <= period_start:
            raise ValueError("period_end must be after period_start")
        return v

    @field_validator("ch4_captured_tonnes")
    @classmethod
    def validate_captured_not_exceeding_generated(
        cls, v: Decimal, info: Any
    ) -> Decimal:
        """Validate captured CH4 does not exceed generated CH4."""
        data = info.data if hasattr(info, "data") else {}
        generated = data.get("ch4_generated_tonnes")
        if generated is not None and v > generated:
            raise ValueError(
                f"ch4_captured_tonnes ({v}) cannot exceed "
                f"ch4_generated_tonnes ({generated})"
            )
        return v


class EnergyRecoveryRecord(BaseModel):
    """Record of energy recovery and grid displacement offsets.

    Tracks energy generated from waste treatment (WtE) and the
    corresponding avoided emissions from displacing grid electricity
    or fossil fuel heat.

    Attributes:
        id: Unique record identifier (UUID).
        facility_id: Reference to the treatment facility.
        calculation_id: Reference to the emission calculation.
        period_start: Start of the recording period.
        period_end: End of the recording period.
        waste_treated_tonnes: Waste mass treated (tonnes).
        ncv_gj_per_tonne: Net calorific value applied (GJ/tonne).
        thermal_energy_gj: Gross thermal energy from waste (GJ).
        electricity_generated_mwh: Electricity generated (MWh).
        heat_recovered_gj: Heat recovered for district heating (GJ).
        grid_ef_tco2e_per_mwh: Grid emission factor (tCO2e/MWh).
        heat_ef_tco2e_per_gj: Heat displacement EF (tCO2e/GJ).
        electricity_offset_tco2e: Avoided emissions from
            electricity generation (tCO2e).
        heat_offset_tco2e: Avoided emissions from heat
            displacement (tCO2e).
        total_offset_tco2e: Total avoided emissions (tCO2e).
        notes: Optional notes.
        tenant_id: Owning tenant identifier.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID)",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the treatment facility",
    )
    calculation_id: str = Field(
        default="",
        description="Reference to the emission calculation",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the recording period",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the recording period",
    )
    waste_treated_tonnes: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Waste mass treated (tonnes)",
    )
    ncv_gj_per_tonne: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Net calorific value applied (GJ/tonne)",
    )
    thermal_energy_gj: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Gross thermal energy from waste (GJ)",
    )
    electricity_generated_mwh: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Electricity generated (MWh)",
    )
    heat_recovered_gj: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Heat recovered for district heating (GJ)",
    )
    grid_ef_tco2e_per_mwh: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Grid emission factor (tCO2e/MWh)",
    )
    heat_ef_tco2e_per_gj: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Heat displacement EF (tCO2e/GJ)",
    )
    electricity_offset_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Avoided emissions from electricity (tCO2e)",
    )
    heat_offset_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Avoided emissions from heat (tCO2e)",
    )
    total_offset_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total avoided emissions (tCO2e)",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes about energy recovery",
    )
    tenant_id: str = Field(
        default="",
        description="Owning tenant identifier",
    )


class CalculationRequest(BaseModel):
    """Request for a waste treatment emission calculation.

    Specifies all parameters needed to compute GHG emissions
    from a waste treatment event including waste quantity,
    treatment method, composition, and method-specific inputs.

    Attributes:
        id: Unique request identifier (UUID).
        facility_id: Reference to the treatment facility.
        waste_stream_id: Optional reference to a registered stream.
        waste_category: Primary waste category.
        treatment_method: Treatment method applied.
        waste_mass_tonnes: Mass of waste treated (tonnes, wet basis).
        composition: Optional detailed waste composition.
        calculation_method: Methodology for emission calculation.
        gwp_source: GWP source for CO2e conversion.
        data_quality_tier: Data quality tier of input data.
        scope: GHG Protocol emission scope.
        climate_zone: Climate zone for decay rate selection.
        biological_input: Additional inputs for biological treatment.
        thermal_input: Additional inputs for thermal treatment.
        wastewater_input: Additional inputs for wastewater treatment.
        include_biogenic_co2: Whether to report biogenic CO2.
        include_energy_offset: Whether to calculate energy offset.
        reference_year: Reference year for the calculation.
        tenant_id: Owning tenant identifier.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier (UUID)",
    )
    facility_id: str = Field(
        default="",
        description="Reference to the treatment facility",
    )
    waste_stream_id: Optional[str] = Field(
        default=None,
        description="Optional reference to a registered waste stream",
    )
    waste_category: WasteCategory = Field(
        ...,
        description="Primary waste category",
    )
    treatment_method: TreatmentMethod = Field(
        ...,
        description="Treatment method applied",
    )
    waste_mass_tonnes: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Mass of waste treated (tonnes, wet basis)",
    )
    composition: Optional[WasteComposition] = Field(
        default=None,
        description="Optional detailed waste composition",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.IPCC_TIER_1,
        description="Methodology for emission calculation",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source for CO2e conversion",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier of input data",
    )
    scope: EmissionScope = Field(
        default=EmissionScope.SCOPE_1,
        description="GHG Protocol emission scope",
    )
    climate_zone: ClimateZone = Field(
        default=ClimateZone.TEMPERATE,
        description="Climate zone for decay rate selection",
    )
    biological_input: Optional[BiologicalTreatmentInput] = Field(
        default=None,
        description="Additional inputs for biological treatment",
    )
    thermal_input: Optional[ThermalTreatmentInput] = Field(
        default=None,
        description="Additional inputs for thermal treatment",
    )
    wastewater_input: Optional[WastewaterTreatmentInput] = Field(
        default=None,
        description="Additional inputs for wastewater treatment",
    )
    include_biogenic_co2: bool = Field(
        default=False,
        description="Whether to report biogenic CO2 separately",
    )
    include_energy_offset: bool = Field(
        default=False,
        description="Whether to calculate energy recovery offset",
    )
    reference_year: int = Field(
        default=2025,
        ge=1900,
        le=2100,
        description="Reference year for the calculation",
    )
    tenant_id: str = Field(
        default="",
        description="Owning tenant identifier",
    )


class GasEmissionDetail(BaseModel):
    """Detailed emission result for a single greenhouse gas.

    Represents the emission of a single gas from a waste treatment
    calculation, including both mass and CO2-equivalent values.

    Attributes:
        gas: Greenhouse gas species.
        emission_mass_tonnes: Emission in tonnes of the gas.
        emission_tco2e: Emission in tonnes CO2 equivalent.
        gwp_applied: GWP value applied for CO2e conversion.
        is_biogenic: Whether emission is biogenic origin.
        is_fossil: Whether emission is fossil origin.
        factor_used: Emission factor value used.
        factor_unit: Unit of the emission factor used.
        formula: Textual description of the calculation formula.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species",
    )
    emission_mass_tonnes: Decimal = Field(
        ...,
        description="Emission in tonnes of the gas",
    )
    emission_tco2e: Decimal = Field(
        ...,
        description="Emission in tonnes CO2 equivalent",
    )
    gwp_applied: Decimal = Field(
        default=Decimal("1"),
        description="GWP value applied for CO2e conversion",
    )
    is_biogenic: bool = Field(
        default=False,
        description="Whether emission is biogenic origin",
    )
    is_fossil: bool = Field(
        default=True,
        description="Whether emission is fossil origin",
    )
    factor_used: Decimal = Field(
        default=Decimal("0"),
        description="Emission factor value used",
    )
    factor_unit: str = Field(
        default="",
        max_length=100,
        description="Unit of the emission factor used",
    )
    formula: str = Field(
        default="",
        max_length=1000,
        description="Textual description of the calculation formula",
    )


class CalculationResult(BaseModel):
    """Complete result of a waste treatment emission calculation.

    Attributes:
        id: Unique result identifier (UUID).
        request_id: Reference to the original calculation request.
        total_co2e: Total emissions in tonnes CO2 equivalent.
        fossil_co2e: Fossil-origin emissions in tCO2e.
        biogenic_co2e: Biogenic-origin CO2 emissions in tCO2e.
        emissions_by_gas: Emissions breakdown by gas (tCO2e).
        gas_details: List of detailed per-gas emission results.
        energy_offset_tco2e: Avoided emissions from energy recovery.
        net_co2e: Net emissions (total minus offset) in tCO2e.
        waste_category: Waste category processed.
        treatment_method: Treatment method applied.
        calculation_method: Calculation method used.
        data_quality_tier: Data quality tier used.
        scope: GHG Protocol scope classification.
        trace_steps: Ordered list of calculation trace steps.
        timestamp: UTC timestamp of calculation completion.
        provenance_hash: SHA-256 provenance hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier (UUID)",
    )
    request_id: str = Field(
        default="",
        description="Reference to the original calculation request",
    )
    total_co2e: Decimal = Field(
        ...,
        description="Total emissions in tonnes CO2 equivalent",
    )
    fossil_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Fossil-origin emissions in tCO2e",
    )
    biogenic_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Biogenic-origin CO2 emissions in tCO2e",
    )
    emissions_by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by gas (tCO2e)",
    )
    gas_details: List[GasEmissionDetail] = Field(
        default_factory=list,
        description="Detailed per-gas emission results",
    )
    energy_offset_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Avoided emissions from energy recovery (tCO2e)",
    )
    net_co2e: Decimal = Field(
        ...,
        description="Net emissions (total minus offset) in tCO2e",
    )
    waste_category: WasteCategory = Field(
        ...,
        description="Waste category processed",
    )
    treatment_method: TreatmentMethod = Field(
        ...,
        description="Treatment method applied",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Calculation method used",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier used",
    )
    scope: EmissionScope = Field(
        default=EmissionScope.SCOPE_1,
        description="GHG Protocol scope classification",
    )
    trace_steps: List[str] = Field(
        default_factory=list,
        description="Ordered list of calculation trace steps",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of calculation completion",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash for audit trail",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )

    @field_validator("trace_steps")
    @classmethod
    def validate_trace_steps(cls, v: List[str]) -> List[str]:
        """Validate trace steps do not exceed maximum."""
        if len(v) > MAX_TRACE_STEPS:
            raise ValueError(
                f"Maximum {MAX_TRACE_STEPS} trace steps allowed, "
                f"got {len(v)}"
            )
        return v

    @field_validator("gas_details")
    @classmethod
    def validate_gas_details(
        cls, v: List[GasEmissionDetail]
    ) -> List[GasEmissionDetail]:
        """Validate gas detail entries do not exceed maximum."""
        if len(v) > MAX_GASES_PER_RESULT:
            raise ValueError(
                f"Maximum {MAX_GASES_PER_RESULT} gas entries allowed, "
                f"got {len(v)}"
            )
        return v


class BatchCalculationRequest(BaseModel):
    """Batch request for multiple waste treatment emission calculations.

    Attributes:
        id: Unique batch request identifier (UUID).
        calculations: List of individual calculation requests.
        gwp_source: GWP source applied to all calculations.
        tenant_id: Owning tenant identifier.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch request identifier (UUID)",
    )
    calculations: List[CalculationRequest] = Field(
        ...,
        min_length=1,
        description="List of individual calculation requests",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source applied to all calculations",
    )
    tenant_id: str = Field(
        default="",
        description="Owning tenant identifier",
    )

    @field_validator("calculations")
    @classmethod
    def validate_calculations(
        cls, v: List[CalculationRequest]
    ) -> List[CalculationRequest]:
        """Validate batch size does not exceed maximum."""
        if len(v) > MAX_CALCULATIONS_PER_BATCH:
            raise ValueError(
                f"Maximum {MAX_CALCULATIONS_PER_BATCH} calculations "
                f"per batch, got {len(v)}"
            )
        return v


class BatchCalculationResult(BaseModel):
    """Result of a batch waste treatment emission calculation.

    Attributes:
        id: Unique batch result identifier (UUID).
        results: List of individual calculation results.
        total_co2e: Aggregate emissions in tonnes CO2e.
        total_fossil_co2e: Aggregate fossil emissions in tCO2e.
        total_biogenic_co2e: Aggregate biogenic CO2 in tCO2e.
        total_energy_offset: Aggregate energy offset in tCO2e.
        net_co2e: Aggregate net emissions in tonnes CO2e.
        calculation_count: Number of calculations in the batch.
        failed_count: Number of calculations that failed.
        emissions_by_method: Emissions aggregated by treatment method.
        emissions_by_waste_type: Emissions aggregated by waste type.
        timestamp: UTC timestamp of batch completion.
        processing_time_ms: Total processing duration in ms.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch result identifier (UUID)",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    total_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Aggregate emissions in tonnes CO2e",
    )
    total_fossil_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Aggregate fossil emissions in tCO2e",
    )
    total_biogenic_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Aggregate biogenic CO2 in tCO2e",
    )
    total_energy_offset: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate energy recovery offset in tCO2e",
    )
    net_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Aggregate net emissions in tonnes CO2e",
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations in the batch",
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations that failed",
    )
    emissions_by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions aggregated by treatment method (tCO2e)",
    )
    emissions_by_waste_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions aggregated by waste type (tCO2e)",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of batch completion",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )


class ComplianceCheckResult(BaseModel):
    """Result of a regulatory compliance check.

    Evaluates a calculation or facility against one of the seven
    supported regulatory frameworks (IPCC, GHG Protocol, CSRD,
    EPA, EU IED, DEFRA, ISO 14064).

    Attributes:
        id: Unique result identifier (UUID).
        framework: Regulatory framework checked.
        status: Overall compliance status.
        total_requirements: Total number of requirements checked.
        passed: Number of requirements passed.
        failed: Number of requirements failed.
        warnings: Number of advisory warnings.
        findings: List of finding descriptions.
        recommendations: List of remediation recommendations.
        calculation_id: Reference to the calculation checked.
        facility_id: Reference to the facility checked.
        checked_at: UTC timestamp of the compliance check.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier (UUID)",
    )
    framework: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Regulatory framework checked",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status",
    )
    total_requirements: int = Field(
        default=0,
        ge=0,
        description="Total number of requirements checked",
    )
    passed: int = Field(
        default=0,
        ge=0,
        description="Number of requirements passed",
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Number of requirements failed",
    )
    warnings: int = Field(
        default=0,
        ge=0,
        description="Number of advisory warnings",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="List of finding descriptions",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of remediation recommendations",
    )
    calculation_id: str = Field(
        default="",
        description="Reference to the calculation checked",
    )
    facility_id: str = Field(
        default="",
        description="Reference to the facility checked",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the compliance check",
    )


class UncertaintyRequest(BaseModel):
    """Request for uncertainty quantification of a calculation.

    Uses Monte Carlo simulation to propagate uncertainty through
    waste composition variability, emission factor ranges, and
    measurement uncertainty.

    Attributes:
        id: Unique request identifier (UUID).
        calculation_id: Reference to the calculation result.
        iterations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.
        confidence_level: Confidence level percentage (e.g. 95.0).
        ef_uncertainty_pct: Emission factor uncertainty as
            percentage (used if no factor-specific range).
        composition_uncertainty_pct: Waste composition uncertainty
            as percentage.
        mass_uncertainty_pct: Waste mass measurement uncertainty
            as percentage.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier (UUID)",
    )
    calculation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the calculation result",
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
        description="Random seed for reproducibility",
    )
    confidence_level: Decimal = Field(
        default=Decimal("95.0"),
        gt=Decimal("0"),
        lt=Decimal("100"),
        description="Confidence level percentage",
    )
    ef_uncertainty_pct: Decimal = Field(
        default=Decimal("50.0"),
        ge=Decimal("0"),
        le=Decimal("200"),
        description="Emission factor uncertainty percentage",
    )
    composition_uncertainty_pct: Decimal = Field(
        default=Decimal("20.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Waste composition uncertainty percentage",
    )
    mass_uncertainty_pct: Decimal = Field(
        default=Decimal("10.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Waste mass measurement uncertainty percentage",
    )


class UncertaintyResult(BaseModel):
    """Result of uncertainty quantification analysis.

    Attributes:
        id: Unique result identifier (UUID).
        calculation_id: Reference to the calculation result.
        mean_co2e: Mean emission estimate in tCO2e.
        std_dev: Standard deviation in tCO2e.
        ci_lower: Lower confidence interval bound in tCO2e.
        ci_upper: Upper confidence interval bound in tCO2e.
        percentiles: Dictionary of percentile values.
        iterations: Number of Monte Carlo iterations performed.
        confidence_level: Confidence level percentage used.
        coefficient_of_variation: CV as a percentage.
        data_quality_score: Overall data quality indicator (1-5).
        timestamp: UTC timestamp of analysis completion.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier (UUID)",
    )
    calculation_id: str = Field(
        default="",
        description="Reference to the calculation result",
    )
    mean_co2e: Decimal = Field(
        ...,
        description="Mean emission estimate in tCO2e",
    )
    std_dev: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Standard deviation in tCO2e",
    )
    ci_lower: Decimal = Field(
        ...,
        description="Lower confidence interval bound in tCO2e",
    )
    ci_upper: Decimal = Field(
        ...,
        description="Upper confidence interval bound in tCO2e",
    )
    percentiles: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Percentile values (e.g. {'5': ..., '95': ...})",
    )
    iterations: int = Field(
        default=0,
        ge=0,
        description="Number of Monte Carlo iterations performed",
    )
    confidence_level: Decimal = Field(
        default=Decimal("95.0"),
        description="Confidence level percentage used",
    )
    coefficient_of_variation: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Coefficient of variation as a percentage",
    )
    data_quality_score: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("1"),
        le=Decimal("5"),
        description="Overall data quality indicator (1-5, 1=best)",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of analysis completion",
    )


class AggregationRequest(BaseModel):
    """Request for aggregating waste treatment emission results.

    Attributes:
        id: Unique request identifier (UUID).
        tenant_id: Tenant identifier for scoping.
        period: Reporting period granularity.
        group_by: Fields to group results by.
        date_from: Start date for the aggregation window.
        date_to: End date for the aggregation window.
        treatment_methods: Optional filter by treatment methods.
        waste_categories: Optional filter by waste categories.
        facility_ids: Optional filter by facility IDs.
        scopes: Optional filter by emission scopes.
        include_offsets: Whether to include energy offsets.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier (UUID)",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Tenant identifier for scoping",
    )
    period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Reporting period granularity",
    )
    group_by: List[str] = Field(
        default_factory=lambda: ["treatment_method"],
        description="Fields to group results by",
    )
    date_from: Optional[datetime] = Field(
        default=None,
        description="Start date for the aggregation window",
    )
    date_to: Optional[datetime] = Field(
        default=None,
        description="End date for the aggregation window",
    )
    treatment_methods: Optional[List[TreatmentMethod]] = Field(
        default=None,
        description="Optional filter by treatment methods",
    )
    waste_categories: Optional[List[WasteCategory]] = Field(
        default=None,
        description="Optional filter by waste categories",
    )
    facility_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional filter by facility IDs",
    )
    scopes: Optional[List[EmissionScope]] = Field(
        default=None,
        description="Optional filter by emission scopes",
    )
    include_offsets: bool = Field(
        default=True,
        description="Whether to include energy recovery offsets",
    )


class AggregationResult(BaseModel):
    """Result of a waste treatment emission aggregation.

    Attributes:
        id: Unique result identifier (UUID).
        groups: Dictionary mapping group keys to aggregated values.
        total_co2e: Total emissions in tonnes CO2e.
        total_fossil_co2e: Total fossil emissions in tCO2e.
        total_biogenic_co2e: Total biogenic CO2 in tCO2e.
        total_energy_offset: Total energy offset in tCO2e.
        net_co2e: Net emissions in tonnes CO2e.
        total_waste_tonnes: Total waste mass processed in tonnes.
        calculation_count: Number of calculations aggregated.
        period: Reporting period used.
        date_from: Start date of the aggregation window.
        date_to: End date of the aggregation window.
        timestamp: UTC timestamp of aggregation completion.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier (UUID)",
    )
    groups: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description="Group keys mapped to aggregated values",
    )
    total_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Total emissions in tonnes CO2e",
    )
    total_fossil_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Total fossil emissions in tCO2e",
    )
    total_biogenic_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Total biogenic CO2 in tCO2e",
    )
    total_energy_offset: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total energy recovery offset in tCO2e",
    )
    net_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Net emissions in tonnes CO2e",
    )
    total_waste_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total waste mass processed in tonnes",
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations aggregated",
    )
    period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Reporting period used",
    )
    date_from: Optional[datetime] = Field(
        default=None,
        description="Start date of the aggregation window",
    )
    date_to: Optional[datetime] = Field(
        default=None,
        description="End date of the aggregation window",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of aggregation completion",
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "WasteCategory",
    "TreatmentMethod",
    "CompostingType",
    "IncineratorType",
    "WastewaterSystem",
    "CalculationMethod",
    "EmissionGas",
    "GWPSource",
    "EmissionFactorSource",
    "DataQualityTier",
    "FacilityType",
    "BiogasComponent",
    "ClimateZone",
    "ComplianceStatus",
    "ReportingPeriod",
    "EmissionScope",
    # Constant tables
    "GWP_VALUES",
    "CONVERSION_FACTOR_CO2_C",
    "CH4_C_RATIO",
    "N2O_N_RATIO",
    "CH4_DENSITY_STP",
    "IPCC_DOC_VALUES",
    "IPCC_MCF_VALUES",
    "IPCC_CARBON_CONTENT",
    "IPCC_COMPOSTING_EF",
    "IPCC_INCINERATION_EF",
    "IPCC_WASTEWATER_MCF",
    "WASTEWATER_BO",
    "WASTEWATER_N2O_EF",
    "INCINERATION_NCV",
    "HALF_LIFE_VALUES",
    "WASTE_DEGRADABILITY_CLASS",
    "OPEN_BURNING_EF",
    "BMP_DEFAULTS",
    "VS_FRACTION",
    "BIOGAS_CH4_FRACTION",
    "ADVANCED_THERMAL_EF",
    # Data models
    "WasteComposition",
    "TreatmentFacilityInfo",
    "WasteStreamInfo",
    "EmissionFactorRecord",
    "BiologicalTreatmentInput",
    "ThermalTreatmentInput",
    "WastewaterTreatmentInput",
    "MethaneRecoveryRecord",
    "EnergyRecoveryRecord",
    "CalculationRequest",
    "GasEmissionDetail",
    "CalculationResult",
    "BatchCalculationRequest",
    "BatchCalculationResult",
    "ComplianceCheckResult",
    "UncertaintyRequest",
    "UncertaintyResult",
    "AggregationRequest",
    "AggregationResult",
    # Scalar constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_STREAMS_PER_CALC",
    "MAX_FACILITIES_PER_TENANT",
    "DEFAULT_DOCf",
    "DEFAULT_F_CH4_LFG",
    "DEFAULT_OXIDATION_FACTOR",
    "DEFAULT_OPEN_BURN_OXIDATION",
    "DEFAULT_FLARE_DESTRUCTION_EFF",
]
