# -*- coding: utf-8 -*-
"""
Land Use Emissions Agent Data Models - AGENT-MRV-006

Pydantic v2 data models for the Land Use Emissions Agent SDK covering
GHG Protocol Scope 1 land use, land-use change, and forestry (LULUCF)
emission calculations including:
- 6 IPCC land categories (forest, cropland, grassland, wetland, settlement,
  other land) with transitions between categories
- 5 carbon pools (above-ground biomass, below-ground biomass, dead wood,
  litter, soil organic carbon)
- 12 IPCC climate zones with associated default biomass and SOC values
- 7 soil types with SOC reference stocks per IPCC 2006 Guidelines
- Stock-difference and gain-loss calculation methods at Tier 1/2/3
- SOC assessment with land use, management, and input factors
- Fire and disturbance emissions (CO2, CH4, N2O, CO)
- Peatland emissions (natural, drained, rewetted, extracted)
- Monte Carlo uncertainty quantification
- Multi-framework regulatory compliance (GHG Protocol, IPCC, CSRD,
  EU LULUCF, UK SECR, UNFCCC)
- SHA-256 provenance chain for complete audit trails

Enumerations (16):
    - LandCategory, CarbonPool, ClimateZone, SoilType, CalculationTier,
      CalculationMethod, EmissionGas, GWPSource, EmissionFactorSource,
      TransitionType, DisturbanceType, PeatlandStatus, ManagementPractice,
      InputLevel, ComplianceStatus, ReportingPeriod

Constants:
    - GWP_VALUES: IPCC AR4/AR5/AR6 GWP values (Decimal)
    - IPCC_AGB_DEFAULTS: Above-ground biomass defaults by land/climate
    - ROOT_SHOOT_RATIOS: Below-ground to above-ground biomass ratios
    - DEAD_WOOD_FRACTION: Dead wood fraction of AGB by climate zone
    - LITTER_STOCKS: Litter carbon stocks by land/climate
    - SOC_REFERENCE_STOCKS: SOC reference stocks by climate/soil
    - SOC_LAND_USE_FACTORS: Land use factors (F_LU)
    - SOC_MANAGEMENT_FACTORS: Management factors (F_MG)
    - SOC_INPUT_FACTORS: Input factors (F_I)
    - BIOMASS_GROWTH_RATES: Annual biomass increment
    - CARBON_FRACTION: Default carbon fraction of dry matter
    - COMBUSTION_FACTORS: Combustion completeness by land/disturbance
    - FIRE_EMISSION_FACTORS: Fire emission factors per gas
    - PEATLAND_EF: Peatland emission factors by status/climate
    - N2O_SOIL_EF: IPCC default EF1 for direct soil N2O
    - CONVERSION_FACTOR_CO2_C: 44/12 molecular weight ratio

Data Models (16):
    - LandParcelInfo, CarbonStockSnapshot, EmissionFactorRecord,
      LandUseTransitionRecord, CalculationRequest, CalculationResult,
      CalculationDetailResult, BatchCalculationRequest,
      BatchCalculationResult, SOCAssessmentRequest, SOCAssessmentResult,
      UncertaintyRequest, UncertaintyResult, ComplianceCheckResult,
      AggregationRequest, AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
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

#: Maximum number of carbon pools in a single calculation request.
MAX_POOLS_PER_CALC: int = 5

#: Maximum number of parcels per tenant.
MAX_PARCELS_PER_TENANT: int = 50_000

#: Default transition period for land-use conversion (years).
DEFAULT_TRANSITION_YEARS: int = 20


# =============================================================================
# Enumerations (16)
# =============================================================================


class LandCategory(str, Enum):
    """IPCC land-use categories for LULUCF reporting.

    Six mutually exclusive land categories as defined in IPCC 2006
    Guidelines Volume 4 (Agriculture, Forestry and Other Land Use).
    Every parcel of managed land is assigned to exactly one category
    at any point in time.

    FOREST_LAND: Land spanning more than 0.5 hectares with trees
        higher than 5 metres and a canopy cover of more than 10 percent,
        or trees able to reach these thresholds in situ.
    CROPLAND: Arable and tillage land, rice paddies, and agroforestry
        systems where vegetation falls below the forest land threshold.
    GRASSLAND: Rangelands, pasture, and other land predominantly
        covered by grasses, forbs, or shrubs below the forest threshold.
    WETLAND: Land covered or saturated by water for all or part of
        the year, including peatlands and managed water bodies.
    SETTLEMENT: Built-up land including transportation infrastructure,
        human settlements, and associated vegetated areas.
    OTHER_LAND: Bare soil, rock, ice, and other land not classified
        in any of the preceding categories.
    """

    FOREST_LAND = "forest_land"
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    SETTLEMENT = "settlement"
    OTHER_LAND = "other_land"


class CarbonPool(str, Enum):
    """IPCC carbon pools tracked in LULUCF calculations.

    Five carbon pools as defined in IPCC 2006 Guidelines Volume 4
    Chapter 1. Each pool is tracked independently and summed for
    total carbon stock change estimates.

    ABOVE_GROUND_BIOMASS: All living biomass above the soil including
        stem, stump, branches, bark, seeds, and foliage.
    BELOW_GROUND_BIOMASS: All living biomass of live roots, excluding
        fine roots less than 2 mm diameter.
    DEAD_WOOD: All non-living woody biomass not contained in litter,
        either standing, lying on the ground, or in the soil.
    LITTER: All non-living biomass with a diameter less than a
        minimum diameter (typically 10 cm), lying dead above the
        mineral or organic soil.
    SOIL_ORGANIC_CARBON: Organic carbon in mineral and organic soils
        to a specified depth (default 30 cm).
    """

    ABOVE_GROUND_BIOMASS = "above_ground_biomass"
    BELOW_GROUND_BIOMASS = "below_ground_biomass"
    DEAD_WOOD = "dead_wood"
    LITTER = "litter"
    SOIL_ORGANIC_CARBON = "soil_organic_carbon"


class ClimateZone(str, Enum):
    """IPCC climate zones for emission factor stratification.

    Twelve climate zones used by IPCC 2006 Guidelines to stratify
    default emission factors, carbon stock values, and biomass
    growth rates. Based on the Koeppen-Geiger climate classification
    adapted for GHG inventory purposes.

    TROPICAL_WET: Tropical wet (Af) - no dry season, mean monthly
        precipitation >= 60 mm in all months.
    TROPICAL_MOIST: Tropical moist (Am) - short dry season,
        annual precipitation >= 1000 mm.
    TROPICAL_DRY: Tropical dry (Aw/As) - distinct dry season,
        annual precipitation < 1000 mm.
    TROPICAL_MONTANE: Tropical montane - tropical regions above
        1000 m elevation with cooler temperatures.
    WARM_TEMPERATE_MOIST: Warm temperate moist (Cfa/Cwa) - mean
        temperature of coldest month between 0 and 18 degrees C.
    WARM_TEMPERATE_DRY: Warm temperate dry (Csb/Csa) - Mediterranean
        climates with dry summers.
    COOL_TEMPERATE_MOIST: Cool temperate moist (Dfb/Dwb) - mean
        temperature of coldest month below 0 degrees C.
    COOL_TEMPERATE_DRY: Cool temperate dry (Dfa/Dwa/BSk) - continental
        dry climates.
    BOREAL_MOIST: Boreal moist (Dfc/Dwc) - long cold winters with
        adequate precipitation.
    BOREAL_DRY: Boreal dry (Dfd/Dwd) - long cold winters with
        limited precipitation.
    POLAR_MOIST: Polar moist (ET) - tundra with adequate moisture.
    POLAR_DRY: Polar dry (EF) - ice cap and very cold dry regions.
    """

    TROPICAL_WET = "tropical_wet"
    TROPICAL_MOIST = "tropical_moist"
    TROPICAL_DRY = "tropical_dry"
    TROPICAL_MONTANE = "tropical_montane"
    WARM_TEMPERATE_MOIST = "warm_temperate_moist"
    WARM_TEMPERATE_DRY = "warm_temperate_dry"
    COOL_TEMPERATE_MOIST = "cool_temperate_moist"
    COOL_TEMPERATE_DRY = "cool_temperate_dry"
    BOREAL_MOIST = "boreal_moist"
    BOREAL_DRY = "boreal_dry"
    POLAR_MOIST = "polar_moist"
    POLAR_DRY = "polar_dry"


class SoilType(str, Enum):
    """IPCC soil types for SOC reference stock stratification.

    Seven soil types used in IPCC 2006 Guidelines Volume 4 Chapter 2
    to stratify soil organic carbon reference stocks and land-use
    change factors.

    HIGH_ACTIVITY_CLAY: Soils with high-activity clay minerals
        (Vertisols, Mollisols, Inceptisols with high CEC).
    LOW_ACTIVITY_CLAY: Soils with low-activity clay minerals
        (Ultisols, Oxisols with low CEC).
    SANDY: Sandy soils with >70% sand and <8% clay
        (Psamments, some Entisols).
    SPODIC: Spodic soils with illuvial accumulation of organic
        matter and aluminum (Spodosols).
    VOLCANIC: Volcanic-ash-derived soils with high organic matter
        retention (Andisols).
    WETLAND_ORGANIC: Organic soils (Histosols) with >20% organic
        carbon content, including peatlands.
    OTHER: Soils not classified in any of the preceding categories.
    """

    HIGH_ACTIVITY_CLAY = "high_activity_clay"
    LOW_ACTIVITY_CLAY = "low_activity_clay"
    SANDY = "sandy"
    SPODIC = "spodic"
    VOLCANIC = "volcanic"
    WETLAND_ORGANIC = "wetland_organic"
    OTHER = "other"


class CalculationTier(str, Enum):
    """IPCC calculation tier levels for methodological complexity.

    TIER_1: Uses global default emission factors and parameters
        from IPCC Guidelines. Simplest approach, highest uncertainty.
    TIER_2: Uses country-specific or region-specific emission factors
        and activity data. Moderate complexity and uncertainty.
    TIER_3: Uses spatially explicit models, repeated measurements,
        and detailed process models. Most complex, lowest uncertainty.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class CalculationMethod(str, Enum):
    """Methodology for calculating carbon stock changes.

    STOCK_DIFFERENCE: Compares total carbon stocks at two points in
        time. Applicable when inventory data (forest inventories,
        soil surveys) are available at both time points.
        Formula: dC = (C_t2 - C_t1) / (t2 - t1)
    GAIN_LOSS: Estimates annual carbon gains (growth, inputs) and
        losses (harvest, disturbance, decomposition) separately.
        Formula: dC = dC_gain - dC_loss
    """

    STOCK_DIFFERENCE = "stock_difference"
    GAIN_LOSS = "gain_loss"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in land use emission calculations.

    CO2: Carbon dioxide - primary gas from carbon stock changes
        in biomass and soils.
    CH4: Methane - emitted from biomass burning, wetlands, rice
        paddies, and peatland decomposition.
    N2O: Nitrous oxide - emitted from soil management, fertilizer
        application, and biomass burning.
    CO: Carbon monoxide - emitted from incomplete combustion during
        biomass burning (precursor gas, not a direct GHG).
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO = "CO"


class GWPSource(str, Enum):
    """IPCC Assessment Report source for Global Warming Potential values.

    IPCC_AR4: Fourth Assessment Report (2007) - 100-year GWP.
    IPCC_AR5: Fifth Assessment Report (2014) - 100-year GWP.
    IPCC_AR6: Sixth Assessment Report (2021) - 100-year GWP.
    IPCC_AR6_GTP: Sixth Assessment Report - Global Temperature
        Potential metric (alternative to GWP).
    """

    IPCC_AR4 = "AR4"
    IPCC_AR5 = "AR5"
    IPCC_AR6 = "AR6"
    IPCC_AR6_GTP = "AR6_GTP"


class EmissionFactorSource(str, Enum):
    """Authoritative source for emission factors and default values.

    IPCC_2006: IPCC 2006 Guidelines for National Greenhouse Gas
        Inventories, Volume 4 (AFOLU).
    IPCC_2019: 2019 Refinement to the 2006 IPCC Guidelines.
    IPCC_WETLANDS_2013: 2013 Supplement to the 2006 IPCC Guidelines:
        Wetlands (peatland emission factors).
    NATIONAL_INVENTORY: Country-specific national inventory factors.
    LITERATURE: Peer-reviewed scientific literature values.
    CUSTOM: User-provided custom emission factors.
    """

    IPCC_2006 = "IPCC_2006"
    IPCC_2019 = "IPCC_2019"
    IPCC_WETLANDS_2013 = "IPCC_WETLANDS_2013"
    NATIONAL_INVENTORY = "NATIONAL_INVENTORY"
    LITERATURE = "LITERATURE"
    CUSTOM = "CUSTOM"


class TransitionType(str, Enum):
    """Type of land-use transition for LULUCF reporting.

    REMAINING: Land remaining in the same IPCC category (e.g. forest
        land remaining forest land). Carbon stock changes are reported
        as annual increments or decrements within the category.
    CONVERSION: Land converted from one IPCC category to another
        (e.g. forest land converted to cropland). Carbon stock
        changes are reported over a default 20-year transition period.
    """

    REMAINING = "remaining"
    CONVERSION = "conversion"


class DisturbanceType(str, Enum):
    """Type of disturbance event affecting carbon stocks.

    FIRE: Wildfire or prescribed burning causing direct biomass
        combustion and emissions of CO2, CH4, N2O, and CO.
    HARVEST: Timber harvesting or other biomass removal for
        commercial or subsistence purposes.
    STORM: Wind damage from storms, hurricanes, or cyclones
        causing tree mortality and biomass transfer to dead wood.
    INSECTS: Insect outbreak causing defoliation, tree mortality,
        and biomass transfer to dead organic matter pools.
    DROUGHT: Extended drought causing tree mortality and reduced
        growth rates.
    FLOOD: Flooding or waterlogging causing tree mortality and
        changes to soil carbon dynamics.
    LAND_CLEARING: Deliberate land clearing for conversion to
        other land uses (deforestation, land preparation).
    NONE: No disturbance event; normal growth and decomposition.
    """

    FIRE = "fire"
    HARVEST = "harvest"
    STORM = "storm"
    INSECTS = "insects"
    DROUGHT = "drought"
    FLOOD = "flood"
    LAND_CLEARING = "land_clearing"
    NONE = "none"


class PeatlandStatus(str, Enum):
    """Management status of peatland areas for emission estimation.

    NATURAL: Undrained peatland in natural or near-natural condition.
        Low CO2 emissions, moderate CH4 emissions.
    DRAINED: Peatland with water table lowered by drainage for
        agriculture, forestry, or peat extraction. High CO2 emissions
        from aerobic decomposition, reduced CH4.
    REWETTED: Previously drained peatland with water table restored.
        Reduced CO2 emissions compared to drained, moderate CH4.
    EXTRACTED: Peatland under active peat extraction (horticultural
        or energy use). Very high CO2 emissions from exposed peat.
    """

    NATURAL = "natural"
    DRAINED = "drained"
    REWETTED = "rewetted"
    EXTRACTED = "extracted"


class ManagementPractice(str, Enum):
    """Soil management practices affecting SOC stocks.

    FULL_TILLAGE: Substantial soil disturbance with full inversion
        or mixing of the soil surface. Conventional plowing.
    REDUCED_TILLAGE: Primary and/or secondary tillage but with
        reduced soil disturbance (chisel plow, ridge tillage).
    NO_TILL: Direct seeding without primary tillage, with only
        minimal soil disturbance at the point of seeding.
    IMPROVED: Improved grassland or forest management with
        practices that increase carbon inputs.
    DEGRADED: Degraded land with management practices that have
        led to loss of soil carbon (overgrazing, erosion).
    NOMINALLY_MANAGED: Default management without specific
        improvement or degradation practices.
    """

    FULL_TILLAGE = "full_tillage"
    REDUCED_TILLAGE = "reduced_tillage"
    NO_TILL = "no_till"
    IMPROVED = "improved"
    DEGRADED = "degraded"
    NOMINALLY_MANAGED = "nominally_managed"


class InputLevel(str, Enum):
    """Level of carbon inputs to soil from residues and amendments.

    LOW: Low residue return, removal of crop residues, bare fallow,
        or low-productivity systems.
    MEDIUM: Representative of annual cropping with residues returned
        to the field (IPCC default/baseline).
    HIGH: High residue return, cover crops, improved varieties with
        higher residue production.
    HIGH_WITH_MANURE: High residue return plus regular application
        of animal manure or compost.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HIGH_WITH_MANURE = "high_with_manure"


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

    MONTHLY: Calendar month aggregation.
    QUARTERLY: Calendar quarter aggregation.
    ANNUAL: Calendar year aggregation (most common for LULUCF).
    CUSTOM: User-defined date range.
    """

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


# =============================================================================
# Constant Tables (all Decimal for deterministic arithmetic)
# =============================================================================


# ---------------------------------------------------------------------------
# GWP values by IPCC Assessment Report
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[GWPSource, Dict[EmissionGas, Decimal]] = {
    GWPSource.IPCC_AR4: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("25"),
        EmissionGas.N2O: Decimal("298"),
        EmissionGas.CO: Decimal("0"),
    },
    GWPSource.IPCC_AR5: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("28"),
        EmissionGas.N2O: Decimal("265"),
        EmissionGas.CO: Decimal("0"),
    },
    GWPSource.IPCC_AR6: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("27.9"),
        EmissionGas.N2O: Decimal("273"),
        EmissionGas.CO: Decimal("0"),
    },
    GWPSource.IPCC_AR6_GTP: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("7.5"),
        EmissionGas.N2O: Decimal("233"),
        EmissionGas.CO: Decimal("0"),
    },
}


# ---------------------------------------------------------------------------
# IPCC default above-ground biomass (AGB) in tC/ha
# ---------------------------------------------------------------------------

IPCC_AGB_DEFAULTS: Dict[Tuple[LandCategory, ClimateZone], Decimal] = {
    # Forest land
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET): Decimal("200"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MOIST): Decimal("180"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_DRY): Decimal("130"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MONTANE): Decimal("150"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("120"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("90"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("100"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("50"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_MOIST): Decimal("40"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_DRY): Decimal("20"),
    (LandCategory.FOREST_LAND, ClimateZone.POLAR_MOIST): Decimal("5"),
    (LandCategory.FOREST_LAND, ClimateZone.POLAR_DRY): Decimal("2"),
    # Cropland
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_WET): Decimal("10"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_MOIST): Decimal("10"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_DRY): Decimal("8"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_MONTANE): Decimal("9"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("8"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("5"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("6"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("4"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_MOIST): Decimal("3"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_DRY): Decimal("2"),
    # Grassland
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_WET): Decimal("16"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_MOIST): Decimal("14"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_DRY): Decimal("8"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_MONTANE): Decimal("10"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("10"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("6"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("8"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("4"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_MOIST): Decimal("5"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_DRY): Decimal("3"),
    # Wetland
    (LandCategory.WETLAND, ClimateZone.TROPICAL_WET): Decimal("120"),
    (LandCategory.WETLAND, ClimateZone.TROPICAL_MOIST): Decimal("100"),
    (LandCategory.WETLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("60"),
    (LandCategory.WETLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("40"),
    (LandCategory.WETLAND, ClimateZone.BOREAL_MOIST): Decimal("25"),
    # Settlement
    (LandCategory.SETTLEMENT, ClimateZone.TROPICAL_WET): Decimal("15"),
    (LandCategory.SETTLEMENT, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("12"),
    (LandCategory.SETTLEMENT, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("10"),
    (LandCategory.SETTLEMENT, ClimateZone.BOREAL_MOIST): Decimal("5"),
    # Other land (minimal biomass)
    (LandCategory.OTHER_LAND, ClimateZone.TROPICAL_WET): Decimal("2"),
    (LandCategory.OTHER_LAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1"),
    (LandCategory.OTHER_LAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1"),
    (LandCategory.OTHER_LAND, ClimateZone.BOREAL_MOIST): Decimal("0.5"),
}


# ---------------------------------------------------------------------------
# Root-to-shoot ratios (BGB/AGB) by land category and climate zone
# ---------------------------------------------------------------------------

ROOT_SHOOT_RATIOS: Dict[Tuple[LandCategory, ClimateZone], Decimal] = {
    # Forest land
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET): Decimal("0.24"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MOIST): Decimal("0.24"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_DRY): Decimal("0.28"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MONTANE): Decimal("0.27"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("0.26"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("0.28"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("0.25"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("0.29"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_MOIST): Decimal("0.24"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_DRY): Decimal("0.24"),
    (LandCategory.FOREST_LAND, ClimateZone.POLAR_MOIST): Decimal("0.40"),
    (LandCategory.FOREST_LAND, ClimateZone.POLAR_DRY): Decimal("0.40"),
    # Cropland (root biomass fraction of above-ground crop biomass)
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_WET): Decimal("0.20"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_MOIST): Decimal("0.20"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_DRY): Decimal("0.20"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("0.22"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("0.22"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("0.22"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("0.22"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_MOIST): Decimal("0.22"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_DRY): Decimal("0.22"),
    # Grassland
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_WET): Decimal("1.60"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_MOIST): Decimal("1.60"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_DRY): Decimal("2.80"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("2.80"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("2.80"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("3.00"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("3.00"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_MOIST): Decimal("4.00"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_DRY): Decimal("4.00"),
}


# ---------------------------------------------------------------------------
# Dead wood fraction of AGB by climate zone
# ---------------------------------------------------------------------------

DEAD_WOOD_FRACTION: Dict[ClimateZone, Decimal] = {
    ClimateZone.TROPICAL_WET: Decimal("0.05"),
    ClimateZone.TROPICAL_MOIST: Decimal("0.06"),
    ClimateZone.TROPICAL_DRY: Decimal("0.08"),
    ClimateZone.TROPICAL_MONTANE: Decimal("0.07"),
    ClimateZone.WARM_TEMPERATE_MOIST: Decimal("0.08"),
    ClimateZone.WARM_TEMPERATE_DRY: Decimal("0.10"),
    ClimateZone.COOL_TEMPERATE_MOIST: Decimal("0.10"),
    ClimateZone.COOL_TEMPERATE_DRY: Decimal("0.12"),
    ClimateZone.BOREAL_MOIST: Decimal("0.12"),
    ClimateZone.BOREAL_DRY: Decimal("0.15"),
    ClimateZone.POLAR_MOIST: Decimal("0.10"),
    ClimateZone.POLAR_DRY: Decimal("0.08"),
}


# ---------------------------------------------------------------------------
# Litter stocks in tC/ha by land category and climate zone
# ---------------------------------------------------------------------------

LITTER_STOCKS: Dict[Tuple[LandCategory, ClimateZone], Decimal] = {
    # Forest land litter stocks
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET): Decimal("2.1"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MOIST): Decimal("2.1"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_DRY): Decimal("3.4"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MONTANE): Decimal("2.8"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("5.0"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("4.5"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("12.0"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("10.0"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_MOIST): Decimal("25.0"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_DRY): Decimal("20.0"),
    (LandCategory.FOREST_LAND, ClimateZone.POLAR_MOIST): Decimal("15.0"),
    (LandCategory.FOREST_LAND, ClimateZone.POLAR_DRY): Decimal("10.0"),
    # Cropland (minimal litter in managed systems)
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_WET): Decimal("0.5"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_MOIST): Decimal("0.5"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_DRY): Decimal("0.3"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("0.8"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("0.5"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.0"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("0.7"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_MOIST): Decimal("0.8"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_DRY): Decimal("0.5"),
    # Grassland
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_WET): Decimal("0.7"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_MOIST): Decimal("0.7"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_DRY): Decimal("0.4"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.2"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("0.8"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("2.0"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.5"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_MOIST): Decimal("3.0"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_DRY): Decimal("2.0"),
}


# ---------------------------------------------------------------------------
# SOC reference stocks in tC/ha for 0-30 cm depth
# Keyed by (ClimateZone, SoilType)
# IPCC 2006 Guidelines Volume 4 Table 2.3
# ---------------------------------------------------------------------------

SOC_REFERENCE_STOCKS: Dict[Tuple[ClimateZone, SoilType], Decimal] = {
    # Tropical wet
    (ClimateZone.TROPICAL_WET, SoilType.HIGH_ACTIVITY_CLAY): Decimal("65"),
    (ClimateZone.TROPICAL_WET, SoilType.LOW_ACTIVITY_CLAY): Decimal("47"),
    (ClimateZone.TROPICAL_WET, SoilType.SANDY): Decimal("39"),
    (ClimateZone.TROPICAL_WET, SoilType.SPODIC): Decimal("66"),
    (ClimateZone.TROPICAL_WET, SoilType.VOLCANIC): Decimal("130"),
    (ClimateZone.TROPICAL_WET, SoilType.WETLAND_ORGANIC): Decimal("86"),
    (ClimateZone.TROPICAL_WET, SoilType.OTHER): Decimal("50"),
    # Tropical moist
    (ClimateZone.TROPICAL_MOIST, SoilType.HIGH_ACTIVITY_CLAY): Decimal("65"),
    (ClimateZone.TROPICAL_MOIST, SoilType.LOW_ACTIVITY_CLAY): Decimal("47"),
    (ClimateZone.TROPICAL_MOIST, SoilType.SANDY): Decimal("39"),
    (ClimateZone.TROPICAL_MOIST, SoilType.SPODIC): Decimal("66"),
    (ClimateZone.TROPICAL_MOIST, SoilType.VOLCANIC): Decimal("130"),
    (ClimateZone.TROPICAL_MOIST, SoilType.WETLAND_ORGANIC): Decimal("86"),
    (ClimateZone.TROPICAL_MOIST, SoilType.OTHER): Decimal("50"),
    # Tropical dry
    (ClimateZone.TROPICAL_DRY, SoilType.HIGH_ACTIVITY_CLAY): Decimal("38"),
    (ClimateZone.TROPICAL_DRY, SoilType.LOW_ACTIVITY_CLAY): Decimal("35"),
    (ClimateZone.TROPICAL_DRY, SoilType.SANDY): Decimal("31"),
    (ClimateZone.TROPICAL_DRY, SoilType.SPODIC): Decimal("40"),
    (ClimateZone.TROPICAL_DRY, SoilType.VOLCANIC): Decimal("70"),
    (ClimateZone.TROPICAL_DRY, SoilType.WETLAND_ORGANIC): Decimal("60"),
    (ClimateZone.TROPICAL_DRY, SoilType.OTHER): Decimal("34"),
    # Tropical montane
    (ClimateZone.TROPICAL_MONTANE, SoilType.HIGH_ACTIVITY_CLAY): Decimal("65"),
    (ClimateZone.TROPICAL_MONTANE, SoilType.LOW_ACTIVITY_CLAY): Decimal("55"),
    (ClimateZone.TROPICAL_MONTANE, SoilType.SANDY): Decimal("45"),
    (ClimateZone.TROPICAL_MONTANE, SoilType.VOLCANIC): Decimal("130"),
    (ClimateZone.TROPICAL_MONTANE, SoilType.OTHER): Decimal("50"),
    # Warm temperate moist
    (ClimateZone.WARM_TEMPERATE_MOIST, SoilType.HIGH_ACTIVITY_CLAY): Decimal("88"),
    (ClimateZone.WARM_TEMPERATE_MOIST, SoilType.LOW_ACTIVITY_CLAY): Decimal("63"),
    (ClimateZone.WARM_TEMPERATE_MOIST, SoilType.SANDY): Decimal("34"),
    (ClimateZone.WARM_TEMPERATE_MOIST, SoilType.SPODIC): Decimal("117"),
    (ClimateZone.WARM_TEMPERATE_MOIST, SoilType.VOLCANIC): Decimal("130"),
    (ClimateZone.WARM_TEMPERATE_MOIST, SoilType.WETLAND_ORGANIC): Decimal("88"),
    (ClimateZone.WARM_TEMPERATE_MOIST, SoilType.OTHER): Decimal("63"),
    # Warm temperate dry
    (ClimateZone.WARM_TEMPERATE_DRY, SoilType.HIGH_ACTIVITY_CLAY): Decimal("56"),
    (ClimateZone.WARM_TEMPERATE_DRY, SoilType.LOW_ACTIVITY_CLAY): Decimal("40"),
    (ClimateZone.WARM_TEMPERATE_DRY, SoilType.SANDY): Decimal("24"),
    (ClimateZone.WARM_TEMPERATE_DRY, SoilType.SPODIC): Decimal("58"),
    (ClimateZone.WARM_TEMPERATE_DRY, SoilType.VOLCANIC): Decimal("80"),
    (ClimateZone.WARM_TEMPERATE_DRY, SoilType.WETLAND_ORGANIC): Decimal("56"),
    (ClimateZone.WARM_TEMPERATE_DRY, SoilType.OTHER): Decimal("40"),
    # Cool temperate moist
    (ClimateZone.COOL_TEMPERATE_MOIST, SoilType.HIGH_ACTIVITY_CLAY): Decimal("95"),
    (ClimateZone.COOL_TEMPERATE_MOIST, SoilType.LOW_ACTIVITY_CLAY): Decimal("85"),
    (ClimateZone.COOL_TEMPERATE_MOIST, SoilType.SANDY): Decimal("71"),
    (ClimateZone.COOL_TEMPERATE_MOIST, SoilType.SPODIC): Decimal("117"),
    (ClimateZone.COOL_TEMPERATE_MOIST, SoilType.VOLCANIC): Decimal("130"),
    (ClimateZone.COOL_TEMPERATE_MOIST, SoilType.WETLAND_ORGANIC): Decimal("95"),
    (ClimateZone.COOL_TEMPERATE_MOIST, SoilType.OTHER): Decimal("80"),
    # Cool temperate dry
    (ClimateZone.COOL_TEMPERATE_DRY, SoilType.HIGH_ACTIVITY_CLAY): Decimal("50"),
    (ClimateZone.COOL_TEMPERATE_DRY, SoilType.LOW_ACTIVITY_CLAY): Decimal("40"),
    (ClimateZone.COOL_TEMPERATE_DRY, SoilType.SANDY): Decimal("34"),
    (ClimateZone.COOL_TEMPERATE_DRY, SoilType.SPODIC): Decimal("55"),
    (ClimateZone.COOL_TEMPERATE_DRY, SoilType.VOLCANIC): Decimal("70"),
    (ClimateZone.COOL_TEMPERATE_DRY, SoilType.WETLAND_ORGANIC): Decimal("50"),
    (ClimateZone.COOL_TEMPERATE_DRY, SoilType.OTHER): Decimal("40"),
    # Boreal moist
    (ClimateZone.BOREAL_MOIST, SoilType.HIGH_ACTIVITY_CLAY): Decimal("68"),
    (ClimateZone.BOREAL_MOIST, SoilType.LOW_ACTIVITY_CLAY): Decimal("68"),
    (ClimateZone.BOREAL_MOIST, SoilType.SANDY): Decimal("10"),
    (ClimateZone.BOREAL_MOIST, SoilType.SPODIC): Decimal("117"),
    (ClimateZone.BOREAL_MOIST, SoilType.VOLCANIC): Decimal("130"),
    (ClimateZone.BOREAL_MOIST, SoilType.WETLAND_ORGANIC): Decimal("68"),
    (ClimateZone.BOREAL_MOIST, SoilType.OTHER): Decimal("68"),
    # Boreal dry
    (ClimateZone.BOREAL_DRY, SoilType.HIGH_ACTIVITY_CLAY): Decimal("50"),
    (ClimateZone.BOREAL_DRY, SoilType.LOW_ACTIVITY_CLAY): Decimal("50"),
    (ClimateZone.BOREAL_DRY, SoilType.SANDY): Decimal("10"),
    (ClimateZone.BOREAL_DRY, SoilType.SPODIC): Decimal("84"),
    (ClimateZone.BOREAL_DRY, SoilType.VOLCANIC): Decimal("80"),
    (ClimateZone.BOREAL_DRY, SoilType.WETLAND_ORGANIC): Decimal("50"),
    (ClimateZone.BOREAL_DRY, SoilType.OTHER): Decimal("40"),
    # Polar moist
    (ClimateZone.POLAR_MOIST, SoilType.HIGH_ACTIVITY_CLAY): Decimal("35"),
    (ClimateZone.POLAR_MOIST, SoilType.SANDY): Decimal("10"),
    (ClimateZone.POLAR_MOIST, SoilType.WETLAND_ORGANIC): Decimal("35"),
    (ClimateZone.POLAR_MOIST, SoilType.OTHER): Decimal("20"),
    # Polar dry
    (ClimateZone.POLAR_DRY, SoilType.HIGH_ACTIVITY_CLAY): Decimal("20"),
    (ClimateZone.POLAR_DRY, SoilType.SANDY): Decimal("5"),
    (ClimateZone.POLAR_DRY, SoilType.WETLAND_ORGANIC): Decimal("20"),
    (ClimateZone.POLAR_DRY, SoilType.OTHER): Decimal("10"),
}


# ---------------------------------------------------------------------------
# SOC land use factors (F_LU) by land category and climate zone
# IPCC 2006 Guidelines Volume 4 Table 5.5
# ---------------------------------------------------------------------------

SOC_LAND_USE_FACTORS: Dict[Tuple[LandCategory, ClimateZone], Decimal] = {
    # Forest land F_LU = 1.0 in all climates (reference)
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MOIST): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_DRY): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MONTANE): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_MOIST): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_DRY): Decimal("1.0"),
    # Cropland - long-term cultivated
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_WET): Decimal("0.58"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_MOIST): Decimal("0.58"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_DRY): Decimal("0.58"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_MONTANE): Decimal("0.64"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("0.69"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("0.69"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("0.69"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("0.69"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_MOIST): Decimal("0.69"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_DRY): Decimal("0.69"),
    # Grassland - nominally managed
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_WET): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_MOIST): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_DRY): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_MOIST): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_DRY): Decimal("1.0"),
    # Settlement
    (LandCategory.SETTLEMENT, ClimateZone.TROPICAL_WET): Decimal("0.50"),
    (LandCategory.SETTLEMENT, ClimateZone.TROPICAL_MOIST): Decimal("0.50"),
    (LandCategory.SETTLEMENT, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("0.55"),
    (LandCategory.SETTLEMENT, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("0.55"),
    (LandCategory.SETTLEMENT, ClimateZone.BOREAL_MOIST): Decimal("0.55"),
    # Other land
    (LandCategory.OTHER_LAND, ClimateZone.TROPICAL_WET): Decimal("0.40"),
    (LandCategory.OTHER_LAND, ClimateZone.TROPICAL_MOIST): Decimal("0.40"),
    (LandCategory.OTHER_LAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("0.45"),
    (LandCategory.OTHER_LAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("0.45"),
    (LandCategory.OTHER_LAND, ClimateZone.BOREAL_MOIST): Decimal("0.45"),
}


# ---------------------------------------------------------------------------
# SOC management factors (F_MG) by management practice and climate zone
# IPCC 2006 Guidelines Volume 4 Table 5.5
# ---------------------------------------------------------------------------

SOC_MANAGEMENT_FACTORS: Dict[Tuple[ManagementPractice, ClimateZone], Decimal] = {
    # Full tillage - reference (F_MG = 1.0)
    (ManagementPractice.FULL_TILLAGE, ClimateZone.TROPICAL_WET): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.TROPICAL_MOIST): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.TROPICAL_DRY): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.TROPICAL_MONTANE): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.BOREAL_MOIST): Decimal("1.0"),
    (ManagementPractice.FULL_TILLAGE, ClimateZone.BOREAL_DRY): Decimal("1.0"),
    # Reduced tillage
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.TROPICAL_WET): Decimal("1.09"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.TROPICAL_MOIST): Decimal("1.09"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.TROPICAL_DRY): Decimal("1.09"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.TROPICAL_MONTANE): Decimal("1.09"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.08"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.08"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.08"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.08"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.BOREAL_MOIST): Decimal("1.08"),
    (ManagementPractice.REDUCED_TILLAGE, ClimateZone.BOREAL_DRY): Decimal("1.08"),
    # No-till
    (ManagementPractice.NO_TILL, ClimateZone.TROPICAL_WET): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.TROPICAL_MOIST): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.TROPICAL_DRY): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.TROPICAL_MONTANE): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.BOREAL_MOIST): Decimal("1.15"),
    (ManagementPractice.NO_TILL, ClimateZone.BOREAL_DRY): Decimal("1.15"),
    # Improved grassland
    (ManagementPractice.IMPROVED, ClimateZone.TROPICAL_WET): Decimal("1.17"),
    (ManagementPractice.IMPROVED, ClimateZone.TROPICAL_MOIST): Decimal("1.17"),
    (ManagementPractice.IMPROVED, ClimateZone.TROPICAL_DRY): Decimal("1.17"),
    (ManagementPractice.IMPROVED, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.14"),
    (ManagementPractice.IMPROVED, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.14"),
    (ManagementPractice.IMPROVED, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.14"),
    (ManagementPractice.IMPROVED, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.14"),
    (ManagementPractice.IMPROVED, ClimateZone.BOREAL_MOIST): Decimal("1.14"),
    (ManagementPractice.IMPROVED, ClimateZone.BOREAL_DRY): Decimal("1.14"),
    # Degraded grassland
    (ManagementPractice.DEGRADED, ClimateZone.TROPICAL_WET): Decimal("0.70"),
    (ManagementPractice.DEGRADED, ClimateZone.TROPICAL_MOIST): Decimal("0.70"),
    (ManagementPractice.DEGRADED, ClimateZone.TROPICAL_DRY): Decimal("0.70"),
    (ManagementPractice.DEGRADED, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("0.95"),
    (ManagementPractice.DEGRADED, ClimateZone.WARM_TEMPERATE_DRY): Decimal("0.95"),
    (ManagementPractice.DEGRADED, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("0.95"),
    (ManagementPractice.DEGRADED, ClimateZone.COOL_TEMPERATE_DRY): Decimal("0.95"),
    (ManagementPractice.DEGRADED, ClimateZone.BOREAL_MOIST): Decimal("0.95"),
    (ManagementPractice.DEGRADED, ClimateZone.BOREAL_DRY): Decimal("0.95"),
    # Nominally managed (default = 1.0)
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.TROPICAL_WET): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.TROPICAL_MOIST): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.TROPICAL_DRY): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.TROPICAL_MONTANE): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.BOREAL_MOIST): Decimal("1.0"),
    (ManagementPractice.NOMINALLY_MANAGED, ClimateZone.BOREAL_DRY): Decimal("1.0"),
}


# ---------------------------------------------------------------------------
# SOC input factors (F_I) by input level
# IPCC 2006 Guidelines Volume 4 Table 5.5
# ---------------------------------------------------------------------------

SOC_INPUT_FACTORS: Dict[InputLevel, Decimal] = {
    InputLevel.LOW: Decimal("0.92"),
    InputLevel.MEDIUM: Decimal("1.0"),
    InputLevel.HIGH: Decimal("1.11"),
    InputLevel.HIGH_WITH_MANURE: Decimal("1.37"),
}


# ---------------------------------------------------------------------------
# Annual biomass growth rates (increment) in tC/ha/yr
# ---------------------------------------------------------------------------

BIOMASS_GROWTH_RATES: Dict[Tuple[LandCategory, ClimateZone], Decimal] = {
    # Forest land
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_WET): Decimal("5.0"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MOIST): Decimal("4.5"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_DRY): Decimal("2.0"),
    (LandCategory.FOREST_LAND, ClimateZone.TROPICAL_MONTANE): Decimal("3.0"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("3.5"),
    (LandCategory.FOREST_LAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("2.0"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("3.0"),
    (LandCategory.FOREST_LAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("1.5"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_MOIST): Decimal("1.0"),
    (LandCategory.FOREST_LAND, ClimateZone.BOREAL_DRY): Decimal("0.5"),
    (LandCategory.FOREST_LAND, ClimateZone.POLAR_MOIST): Decimal("0.2"),
    (LandCategory.FOREST_LAND, ClimateZone.POLAR_DRY): Decimal("0.1"),
    # Grassland
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_WET): Decimal("3.0"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_MOIST): Decimal("2.5"),
    (LandCategory.GRASSLAND, ClimateZone.TROPICAL_DRY): Decimal("1.5"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("2.0"),
    (LandCategory.GRASSLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.0"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.5"),
    (LandCategory.GRASSLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("0.8"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_MOIST): Decimal("0.5"),
    (LandCategory.GRASSLAND, ClimateZone.BOREAL_DRY): Decimal("0.3"),
    # Cropland (annual harvest cycle, not long-term accumulation)
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_WET): Decimal("2.5"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_MOIST): Decimal("2.0"),
    (LandCategory.CROPLAND, ClimateZone.TROPICAL_DRY): Decimal("1.2"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_MOIST): Decimal("1.8"),
    (LandCategory.CROPLAND, ClimateZone.WARM_TEMPERATE_DRY): Decimal("1.0"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_MOIST): Decimal("1.5"),
    (LandCategory.CROPLAND, ClimateZone.COOL_TEMPERATE_DRY): Decimal("0.8"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_MOIST): Decimal("0.5"),
    (LandCategory.CROPLAND, ClimateZone.BOREAL_DRY): Decimal("0.3"),
}


# ---------------------------------------------------------------------------
# Carbon fraction of dry matter (default IPCC value)
# ---------------------------------------------------------------------------

CARBON_FRACTION: Decimal = Decimal("0.47")


# ---------------------------------------------------------------------------
# Combustion factors (fraction of biomass consumed) by land/disturbance
# ---------------------------------------------------------------------------

COMBUSTION_FACTORS: Dict[Tuple[LandCategory, DisturbanceType], Decimal] = {
    # Forest fire combustion completeness
    (LandCategory.FOREST_LAND, DisturbanceType.FIRE): Decimal("0.45"),
    (LandCategory.FOREST_LAND, DisturbanceType.LAND_CLEARING): Decimal("0.60"),
    (LandCategory.FOREST_LAND, DisturbanceType.HARVEST): Decimal("0.0"),
    (LandCategory.FOREST_LAND, DisturbanceType.STORM): Decimal("0.0"),
    (LandCategory.FOREST_LAND, DisturbanceType.INSECTS): Decimal("0.0"),
    (LandCategory.FOREST_LAND, DisturbanceType.NONE): Decimal("0.0"),
    # Grassland fire
    (LandCategory.GRASSLAND, DisturbanceType.FIRE): Decimal("0.74"),
    (LandCategory.GRASSLAND, DisturbanceType.LAND_CLEARING): Decimal("0.80"),
    (LandCategory.GRASSLAND, DisturbanceType.NONE): Decimal("0.0"),
    # Cropland fire (post-harvest burning)
    (LandCategory.CROPLAND, DisturbanceType.FIRE): Decimal("0.80"),
    (LandCategory.CROPLAND, DisturbanceType.LAND_CLEARING): Decimal("0.90"),
    (LandCategory.CROPLAND, DisturbanceType.NONE): Decimal("0.0"),
    # Wetland fire (peatland fires)
    (LandCategory.WETLAND, DisturbanceType.FIRE): Decimal("0.35"),
    (LandCategory.WETLAND, DisturbanceType.NONE): Decimal("0.0"),
    # Settlement
    (LandCategory.SETTLEMENT, DisturbanceType.FIRE): Decimal("0.30"),
    (LandCategory.SETTLEMENT, DisturbanceType.NONE): Decimal("0.0"),
    # Other land
    (LandCategory.OTHER_LAND, DisturbanceType.FIRE): Decimal("0.50"),
    (LandCategory.OTHER_LAND, DisturbanceType.NONE): Decimal("0.0"),
}


# ---------------------------------------------------------------------------
# Fire emission factors in g gas per kg dry matter burned
# IPCC 2006 Guidelines Volume 4 Table 2.5
# ---------------------------------------------------------------------------

FIRE_EMISSION_FACTORS: Dict[EmissionGas, Decimal] = {
    EmissionGas.CO2: Decimal("1580"),
    EmissionGas.CH4: Decimal("6.8"),
    EmissionGas.N2O: Decimal("0.20"),
    EmissionGas.CO: Decimal("104"),
}


# ---------------------------------------------------------------------------
# Peatland emission factors by status and climate zone
# IPCC Wetlands Supplement 2013
# Values: Dict with keys "CO2" (tC/ha/yr), "CH4" (kg CH4/ha/yr),
#         "DOC" (tC/ha/yr dissolved organic carbon export)
# ---------------------------------------------------------------------------

PEATLAND_EF: Dict[
    Tuple[PeatlandStatus, ClimateZone], Dict[str, Decimal]
] = {
    # Tropical drained peatland
    (PeatlandStatus.DRAINED, ClimateZone.TROPICAL_WET): {
        "CO2": Decimal("11.0"),
        "CH4": Decimal("0.0"),
        "DOC": Decimal("0.42"),
    },
    (PeatlandStatus.DRAINED, ClimateZone.TROPICAL_MOIST): {
        "CO2": Decimal("11.0"),
        "CH4": Decimal("0.0"),
        "DOC": Decimal("0.42"),
    },
    (PeatlandStatus.DRAINED, ClimateZone.TROPICAL_DRY): {
        "CO2": Decimal("5.3"),
        "CH4": Decimal("0.0"),
        "DOC": Decimal("0.30"),
    },
    # Temperate drained peatland
    (PeatlandStatus.DRAINED, ClimateZone.WARM_TEMPERATE_MOIST): {
        "CO2": Decimal("5.7"),
        "CH4": Decimal("6.1"),
        "DOC": Decimal("0.31"),
    },
    (PeatlandStatus.DRAINED, ClimateZone.COOL_TEMPERATE_MOIST): {
        "CO2": Decimal("5.7"),
        "CH4": Decimal("6.1"),
        "DOC": Decimal("0.31"),
    },
    # Boreal drained peatland
    (PeatlandStatus.DRAINED, ClimateZone.BOREAL_MOIST): {
        "CO2": Decimal("2.6"),
        "CH4": Decimal("2.5"),
        "DOC": Decimal("0.24"),
    },
    (PeatlandStatus.DRAINED, ClimateZone.BOREAL_DRY): {
        "CO2": Decimal("2.6"),
        "CH4": Decimal("2.5"),
        "DOC": Decimal("0.24"),
    },
    # Natural peatland
    (PeatlandStatus.NATURAL, ClimateZone.TROPICAL_WET): {
        "CO2": Decimal("0.0"),
        "CH4": Decimal("150.0"),
        "DOC": Decimal("0.50"),
    },
    (PeatlandStatus.NATURAL, ClimateZone.TROPICAL_MOIST): {
        "CO2": Decimal("0.0"),
        "CH4": Decimal("150.0"),
        "DOC": Decimal("0.50"),
    },
    (PeatlandStatus.NATURAL, ClimateZone.WARM_TEMPERATE_MOIST): {
        "CO2": Decimal("0.0"),
        "CH4": Decimal("115.0"),
        "DOC": Decimal("0.30"),
    },
    (PeatlandStatus.NATURAL, ClimateZone.COOL_TEMPERATE_MOIST): {
        "CO2": Decimal("0.0"),
        "CH4": Decimal("80.0"),
        "DOC": Decimal("0.30"),
    },
    (PeatlandStatus.NATURAL, ClimateZone.BOREAL_MOIST): {
        "CO2": Decimal("0.0"),
        "CH4": Decimal("30.0"),
        "DOC": Decimal("0.18"),
    },
    (PeatlandStatus.NATURAL, ClimateZone.BOREAL_DRY): {
        "CO2": Decimal("0.0"),
        "CH4": Decimal("10.0"),
        "DOC": Decimal("0.12"),
    },
    # Rewetted peatland
    (PeatlandStatus.REWETTED, ClimateZone.TROPICAL_WET): {
        "CO2": Decimal("2.0"),
        "CH4": Decimal("125.0"),
        "DOC": Decimal("0.45"),
    },
    (PeatlandStatus.REWETTED, ClimateZone.TROPICAL_MOIST): {
        "CO2": Decimal("2.0"),
        "CH4": Decimal("125.0"),
        "DOC": Decimal("0.45"),
    },
    (PeatlandStatus.REWETTED, ClimateZone.WARM_TEMPERATE_MOIST): {
        "CO2": Decimal("1.8"),
        "CH4": Decimal("90.0"),
        "DOC": Decimal("0.28"),
    },
    (PeatlandStatus.REWETTED, ClimateZone.COOL_TEMPERATE_MOIST): {
        "CO2": Decimal("1.5"),
        "CH4": Decimal("65.0"),
        "DOC": Decimal("0.25"),
    },
    (PeatlandStatus.REWETTED, ClimateZone.BOREAL_MOIST): {
        "CO2": Decimal("0.5"),
        "CH4": Decimal("25.0"),
        "DOC": Decimal("0.15"),
    },
    # Extracted peatland
    (PeatlandStatus.EXTRACTED, ClimateZone.TROPICAL_WET): {
        "CO2": Decimal("14.0"),
        "CH4": Decimal("0.0"),
        "DOC": Decimal("0.55"),
    },
    (PeatlandStatus.EXTRACTED, ClimateZone.WARM_TEMPERATE_MOIST): {
        "CO2": Decimal("10.0"),
        "CH4": Decimal("3.0"),
        "DOC": Decimal("0.40"),
    },
    (PeatlandStatus.EXTRACTED, ClimateZone.COOL_TEMPERATE_MOIST): {
        "CO2": Decimal("8.0"),
        "CH4": Decimal("4.0"),
        "DOC": Decimal("0.35"),
    },
    (PeatlandStatus.EXTRACTED, ClimateZone.BOREAL_MOIST): {
        "CO2": Decimal("4.0"),
        "CH4": Decimal("2.0"),
        "DOC": Decimal("0.28"),
    },
}


# ---------------------------------------------------------------------------
# IPCC default N2O emission factor for direct soil emissions (EF1)
# IPCC 2006 Guidelines Volume 4 Chapter 11 Table 11.1
# Fraction of nitrogen input emitted as N2O-N
# ---------------------------------------------------------------------------

N2O_SOIL_EF: Decimal = Decimal("0.01")


# ---------------------------------------------------------------------------
# CO2-to-C conversion factor (molecular weight ratio 44/12)
# ---------------------------------------------------------------------------

CONVERSION_FACTOR_CO2_C: Decimal = Decimal("3.6667")


# =============================================================================
# Pydantic Data Models (16)
# =============================================================================


class LandParcelInfo(BaseModel):
    """Information about a land parcel for LULUCF tracking.

    Represents a single parcel of managed land with its geographic,
    climatic, and soil characteristics. Every parcel must be assigned
    to a tenant and a land-use category.

    Attributes:
        id: Unique parcel identifier (UUID).
        name: Human-readable parcel name or label.
        area_ha: Parcel area in hectares (must be > 0).
        land_category: Current IPCC land-use category.
        climate_zone: IPCC climate zone for the parcel location.
        soil_type: IPCC soil type for SOC assessment.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        tenant_id: Owning tenant identifier for multi-tenancy.
        country_code: ISO 3166-1 alpha-2 country code.
        management_practice: Current soil management practice.
        input_level: Carbon input level for SOC factors.
        peatland_status: Peatland management status (if applicable).
        created_at: UTC timestamp of parcel registration.
        updated_at: UTC timestamp of last update.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique parcel identifier (UUID)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable parcel name",
    )
    area_ha: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Parcel area in hectares",
    )
    land_category: LandCategory = Field(
        ...,
        description="Current IPCC land-use category",
    )
    climate_zone: ClimateZone = Field(
        ...,
        description="IPCC climate zone",
    )
    soil_type: SoilType = Field(
        ...,
        description="IPCC soil type for SOC assessment",
    )
    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="WGS84 latitude in decimal degrees",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="WGS84 longitude in decimal degrees",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Owning tenant identifier",
    )
    country_code: str = Field(
        default="",
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    management_practice: ManagementPractice = Field(
        default=ManagementPractice.NOMINALLY_MANAGED,
        description="Current soil management practice",
    )
    input_level: InputLevel = Field(
        default=InputLevel.MEDIUM,
        description="Carbon input level for SOC factors",
    )
    peatland_status: Optional[PeatlandStatus] = Field(
        default=None,
        description="Peatland management status if applicable",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of parcel registration",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of last update",
    )


class CarbonStockSnapshot(BaseModel):
    """Point-in-time carbon stock measurement for a parcel and pool.

    Represents a single carbon stock observation used in the
    stock-difference method for calculating carbon stock changes.

    Attributes:
        id: Unique snapshot identifier (UUID).
        parcel_id: Reference to the land parcel.
        pool: Carbon pool being measured.
        stock_tc_ha: Carbon stock in tonnes C per hectare.
        measurement_date: Date of the measurement.
        tier: IPCC calculation tier of the measurement.
        source: Source authority for the stock value.
        uncertainty_pct: Measurement uncertainty as a percentage.
        notes: Optional notes about the measurement.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique snapshot identifier (UUID)",
    )
    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the land parcel",
    )
    pool: CarbonPool = Field(
        ...,
        description="Carbon pool being measured",
    )
    stock_tc_ha: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Carbon stock in tonnes C per hectare",
    )
    measurement_date: datetime = Field(
        ...,
        description="Date of the measurement",
    )
    tier: CalculationTier = Field(
        default=CalculationTier.TIER_1,
        description="IPCC calculation tier of the measurement",
    )
    source: EmissionFactorSource = Field(
        default=EmissionFactorSource.IPCC_2006,
        description="Source authority for the stock value",
    )
    uncertainty_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Measurement uncertainty as a percentage",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes about the measurement",
    )


class EmissionFactorRecord(BaseModel):
    """Emission factor record for land use emission calculations.

    Stores a single emission factor with its source, applicability
    scope, and units for use in LULUCF calculations.

    Attributes:
        id: Unique emission factor record identifier (UUID).
        land_category: IPCC land category the factor applies to.
        gas: Greenhouse gas species.
        ef_value: Emission factor value.
        ef_unit: Unit of the emission factor (e.g. tCO2/ha/yr).
        source: Authoritative source of the factor.
        climate_zone: Optional climate zone scoping.
        soil_type: Optional soil type scoping.
        tier: IPCC tier level of the factor.
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
    land_category: LandCategory = Field(
        ...,
        description="IPCC land category the factor applies to",
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
    climate_zone: Optional[ClimateZone] = Field(
        default=None,
        description="Optional climate zone scoping",
    )
    soil_type: Optional[SoilType] = Field(
        default=None,
        description="Optional soil type scoping",
    )
    tier: CalculationTier = Field(
        default=CalculationTier.TIER_1,
        description="IPCC tier level of the factor",
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


class LandUseTransitionRecord(BaseModel):
    """Record of a land-use transition event.

    Tracks a parcel's transition from one IPCC land category to
    another, including the transition date, area affected, and
    transition type (remaining vs conversion).

    Attributes:
        id: Unique transition record identifier (UUID).
        parcel_id: Reference to the land parcel.
        from_category: IPCC land category before transition.
        to_category: IPCC land category after transition.
        transition_date: Date of the transition event.
        area_ha: Area affected by the transition in hectares.
        transition_type: Whether land is remaining or converted.
        disturbance_type: Type of disturbance causing transition.
        notes: Optional notes about the transition.
        created_at: UTC timestamp of record creation.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transition record identifier (UUID)",
    )
    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the land parcel",
    )
    from_category: LandCategory = Field(
        ...,
        description="IPCC land category before transition",
    )
    to_category: LandCategory = Field(
        ...,
        description="IPCC land category after transition",
    )
    transition_date: datetime = Field(
        ...,
        description="Date of the transition event",
    )
    area_ha: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Area affected by transition in hectares",
    )
    transition_type: TransitionType = Field(
        ...,
        description="Whether land is remaining or converted",
    )
    disturbance_type: DisturbanceType = Field(
        default=DisturbanceType.NONE,
        description="Type of disturbance causing transition",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes about the transition",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of record creation",
    )


class CalculationRequest(BaseModel):
    """Request for a land use emission calculation.

    Specifies all parameters needed to compute carbon stock changes
    and resulting GHG emissions for a land parcel or transition event.

    Attributes:
        id: Unique request identifier (UUID).
        parcel_id: Reference to the land parcel.
        from_category: Land category before (for conversions).
        to_category: Land category after (or same for remaining).
        area_ha: Area in hectares for the calculation.
        climate_zone: IPCC climate zone.
        soil_type: IPCC soil type.
        tier: IPCC calculation tier.
        method: Calculation method (stock_difference or gain_loss).
        gwp_source: GWP source for CO2e conversion.
        pools: List of carbon pools to include.
        management_practice: Soil management practice.
        input_level: Carbon input level.
        include_fire: Whether to include fire emissions.
        include_n2o: Whether to include soil N2O emissions.
        include_peatland: Whether to include peatland emissions.
        disturbance_type: Type of disturbance event.
        peatland_status: Peatland status if applicable.
        transition_years: Transition period in years.
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
    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the land parcel",
    )
    from_category: LandCategory = Field(
        ...,
        description="Land category before transition",
    )
    to_category: LandCategory = Field(
        ...,
        description="Land category after transition",
    )
    area_ha: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Area in hectares",
    )
    climate_zone: ClimateZone = Field(
        ...,
        description="IPCC climate zone",
    )
    soil_type: SoilType = Field(
        ...,
        description="IPCC soil type",
    )
    tier: CalculationTier = Field(
        default=CalculationTier.TIER_1,
        description="IPCC calculation tier",
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.STOCK_DIFFERENCE,
        description="Calculation method",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.IPCC_AR6,
        description="GWP source for CO2e conversion",
    )
    pools: List[CarbonPool] = Field(
        default_factory=lambda: [
            CarbonPool.ABOVE_GROUND_BIOMASS,
            CarbonPool.BELOW_GROUND_BIOMASS,
            CarbonPool.DEAD_WOOD,
            CarbonPool.LITTER,
            CarbonPool.SOIL_ORGANIC_CARBON,
        ],
        description="Carbon pools to include in calculation",
    )
    management_practice: ManagementPractice = Field(
        default=ManagementPractice.NOMINALLY_MANAGED,
        description="Soil management practice",
    )
    input_level: InputLevel = Field(
        default=InputLevel.MEDIUM,
        description="Carbon input level",
    )
    include_fire: bool = Field(
        default=False,
        description="Whether to include fire emissions",
    )
    include_n2o: bool = Field(
        default=False,
        description="Whether to include soil N2O emissions",
    )
    include_peatland: bool = Field(
        default=False,
        description="Whether to include peatland emissions",
    )
    disturbance_type: DisturbanceType = Field(
        default=DisturbanceType.NONE,
        description="Type of disturbance event",
    )
    peatland_status: Optional[PeatlandStatus] = Field(
        default=None,
        description="Peatland status if applicable",
    )
    transition_years: int = Field(
        default=DEFAULT_TRANSITION_YEARS,
        gt=0,
        le=100,
        description="Transition period in years",
    )
    reference_year: int = Field(
        default=2024,
        ge=1900,
        le=2100,
        description="Reference year for the calculation",
    )
    tenant_id: str = Field(
        default="",
        description="Owning tenant identifier",
    )

    @field_validator("pools")
    @classmethod
    def validate_pools(cls, v: List[CarbonPool]) -> List[CarbonPool]:
        """Validate that at least one carbon pool is specified."""
        if not v:
            raise ValueError("At least one carbon pool must be specified")
        if len(v) > MAX_POOLS_PER_CALC:
            raise ValueError(
                f"Maximum {MAX_POOLS_PER_CALC} pools allowed, got {len(v)}"
            )
        return v


class CalculationDetailResult(BaseModel):
    """Detailed emission result for a single pool/gas combination.

    Attributes:
        pool: Carbon pool contributing the emission.
        gas: Greenhouse gas species.
        emission_tc: Emission in tonnes of carbon.
        emission_tco2e: Emission in tonnes CO2 equivalent.
        is_removal: True if this represents a carbon removal (sink).
        factor_used: Emission factor or stock value used.
        formula: Textual description of the calculation formula.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    pool: CarbonPool = Field(
        ...,
        description="Carbon pool contributing the emission",
    )
    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species",
    )
    emission_tc: Decimal = Field(
        ...,
        description="Emission in tonnes of carbon",
    )
    emission_tco2e: Decimal = Field(
        ...,
        description="Emission in tonnes CO2 equivalent",
    )
    is_removal: bool = Field(
        default=False,
        description="True if this represents a carbon removal",
    )
    factor_used: Decimal = Field(
        default=Decimal("0"),
        description="Emission factor or stock value used",
    )
    formula: str = Field(
        default="",
        max_length=1000,
        description="Textual description of the calculation formula",
    )


class CalculationResult(BaseModel):
    """Complete result of a land use emission calculation.

    Attributes:
        id: Unique result identifier (UUID).
        request: The original calculation request.
        total_co2e: Total emissions in tonnes CO2 equivalent.
        emissions_by_pool: Emissions breakdown by carbon pool.
        emissions_by_gas: Emissions breakdown by greenhouse gas.
        removals_co2e: Total carbon removals in tonnes CO2e.
        net_co2e: Net emissions (emissions minus removals) in tCO2e.
        tier: IPCC tier used for the calculation.
        method: Calculation method used.
        details: List of detailed per-pool per-gas results.
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
    request: CalculationRequest = Field(
        ...,
        description="The original calculation request",
    )
    total_co2e: Decimal = Field(
        ...,
        description="Total emissions in tonnes CO2 equivalent",
    )
    emissions_by_pool: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by carbon pool (tCO2e)",
    )
    emissions_by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by greenhouse gas (tCO2e)",
    )
    removals_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Total carbon removals in tonnes CO2e",
    )
    net_co2e: Decimal = Field(
        ...,
        description="Net emissions (emissions - removals) in tCO2e",
    )
    tier: CalculationTier = Field(
        ...,
        description="IPCC tier used for the calculation",
    )
    method: CalculationMethod = Field(
        ...,
        description="Calculation method used",
    )
    details: List[CalculationDetailResult] = Field(
        default_factory=list,
        description="Detailed per-pool per-gas results",
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


class BatchCalculationRequest(BaseModel):
    """Batch request for multiple land use emission calculations.

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
        default=GWPSource.IPCC_AR6,
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
    """Result of a batch land use emission calculation.

    Attributes:
        id: Unique batch result identifier (UUID).
        results: List of individual calculation results.
        total_co2e: Aggregate emissions in tonnes CO2e.
        total_removals: Aggregate removals in tonnes CO2e.
        net_co2e: Aggregate net emissions in tonnes CO2e.
        calculation_count: Number of calculations in the batch.
        timestamp: UTC timestamp of batch completion.
        processing_time_ms: Total processing duration in milliseconds.
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
    total_removals: Decimal = Field(
        default=Decimal("0"),
        description="Aggregate removals in tonnes CO2e",
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
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of batch completion",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )


class SOCAssessmentRequest(BaseModel):
    """Request for a soil organic carbon assessment.

    Uses IPCC Tier 1 approach: SOC = SOC_ref * F_LU * F_MG * F_I

    Attributes:
        id: Unique request identifier (UUID).
        parcel_id: Reference to the land parcel.
        climate_zone: IPCC climate zone.
        soil_type: IPCC soil type.
        land_category: Current land-use category.
        management_practice: Soil management practice.
        input_level: Carbon input level.
        depth_cm: Soil assessment depth in centimeters.
        transition_years: Transition period for annualisation.
        previous_land_category: Previous land category (optional).
        previous_management: Previous management practice (optional).
        previous_input_level: Previous input level (optional).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier (UUID)",
    )
    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the land parcel",
    )
    climate_zone: ClimateZone = Field(
        ...,
        description="IPCC climate zone",
    )
    soil_type: SoilType = Field(
        ...,
        description="IPCC soil type",
    )
    land_category: LandCategory = Field(
        ...,
        description="Current land-use category",
    )
    management_practice: ManagementPractice = Field(
        default=ManagementPractice.NOMINALLY_MANAGED,
        description="Soil management practice",
    )
    input_level: InputLevel = Field(
        default=InputLevel.MEDIUM,
        description="Carbon input level",
    )
    depth_cm: int = Field(
        default=30,
        gt=0,
        le=300,
        description="Soil assessment depth in centimeters",
    )
    transition_years: int = Field(
        default=DEFAULT_TRANSITION_YEARS,
        gt=0,
        le=100,
        description="Transition period for annualisation",
    )
    previous_land_category: Optional[LandCategory] = Field(
        default=None,
        description="Previous land category for change calculation",
    )
    previous_management: Optional[ManagementPractice] = Field(
        default=None,
        description="Previous management practice",
    )
    previous_input_level: Optional[InputLevel] = Field(
        default=None,
        description="Previous input level",
    )


class SOCAssessmentResult(BaseModel):
    """Result of a soil organic carbon assessment.

    Attributes:
        id: Unique result identifier (UUID).
        request_id: Reference to the assessment request.
        soc_ref: SOC reference stock in tC/ha.
        f_lu: Land use factor applied.
        f_mg: Management factor applied.
        f_i: Input factor applied.
        soc_current: Current SOC stock in tC/ha (SOC_ref * F_LU * F_MG * F_I).
        soc_previous: Previous SOC stock in tC/ha (if applicable).
        delta_soc_annual: Annual SOC change in tC/ha/yr.
        delta_soc_total: Total SOC change over transition period.
        depth_cm: Depth of the assessment in cm.
        timestamp: UTC timestamp of assessment completion.
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
        description="Reference to the assessment request",
    )
    soc_ref: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="SOC reference stock in tC/ha",
    )
    f_lu: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Land use factor applied",
    )
    f_mg: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Management factor applied",
    )
    f_i: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Input factor applied",
    )
    soc_current: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Current SOC stock in tC/ha",
    )
    soc_previous: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Previous SOC stock in tC/ha",
    )
    delta_soc_annual: Decimal = Field(
        default=Decimal("0"),
        description="Annual SOC change in tC/ha/yr",
    )
    delta_soc_total: Decimal = Field(
        default=Decimal("0"),
        description="Total SOC change over transition period in tC/ha",
    )
    depth_cm: int = Field(
        default=30,
        gt=0,
        description="Depth of the assessment in cm",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of assessment completion",
    )


class UncertaintyRequest(BaseModel):
    """Request for uncertainty quantification of a calculation.

    Attributes:
        id: Unique request identifier (UUID).
        calculation_id: Reference to the calculation result.
        iterations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.
        confidence_level: Confidence level percentage (e.g. 95.0).
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


class UncertaintyResult(BaseModel):
    """Result of uncertainty quantification analysis.

    Attributes:
        id: Unique result identifier (UUID).
        calculation_id: Reference to the calculation result.
        mean_co2e: Mean emission estimate in tCO2e.
        std_dev: Standard deviation in tCO2e.
        ci_lower: Lower confidence interval bound in tCO2e.
        ci_upper: Upper confidence interval bound in tCO2e.
        percentiles: Dictionary of percentile values (e.g. {"5": ..., "95": ...}).
        iterations: Number of Monte Carlo iterations performed.
        confidence_level: Confidence level percentage used.
        coefficient_of_variation: CV as a percentage.
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
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of analysis completion",
    )


class ComplianceCheckResult(BaseModel):
    """Result of a regulatory compliance check.

    Attributes:
        id: Unique result identifier (UUID).
        framework: Regulatory framework checked.
        status: Overall compliance status.
        total_requirements: Total number of requirements checked.
        passed: Number of requirements passed.
        failed: Number of requirements failed.
        findings: List of finding descriptions.
        recommendations: List of remediation recommendations.
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
    findings: List[str] = Field(
        default_factory=list,
        description="List of finding descriptions",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of remediation recommendations",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the compliance check",
    )


class AggregationRequest(BaseModel):
    """Request for aggregating land use emission results.

    Attributes:
        id: Unique request identifier (UUID).
        tenant_id: Tenant identifier for scoping.
        period: Reporting period granularity.
        group_by: Fields to group results by.
        date_from: Start date for the aggregation window.
        date_to: End date for the aggregation window.
        land_categories: Optional filter by land categories.
        climate_zones: Optional filter by climate zones.
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
        default_factory=lambda: ["land_category"],
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
    land_categories: Optional[List[LandCategory]] = Field(
        default=None,
        description="Optional filter by land categories",
    )
    climate_zones: Optional[List[ClimateZone]] = Field(
        default=None,
        description="Optional filter by climate zones",
    )


class AggregationResult(BaseModel):
    """Result of a land use emission aggregation.

    Attributes:
        id: Unique result identifier (UUID).
        groups: Dictionary mapping group keys to aggregated values.
        total_co2e: Total emissions in tonnes CO2e.
        total_removals: Total removals in tonnes CO2e.
        net_co2e: Net emissions in tonnes CO2e.
        area_ha: Total area covered in hectares.
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
    total_removals: Decimal = Field(
        default=Decimal("0"),
        description="Total removals in tonnes CO2e",
    )
    net_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Net emissions in tonnes CO2e",
    )
    area_ha: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total area covered in hectares",
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
    "LandCategory",
    "CarbonPool",
    "ClimateZone",
    "SoilType",
    "CalculationTier",
    "CalculationMethod",
    "EmissionGas",
    "GWPSource",
    "EmissionFactorSource",
    "TransitionType",
    "DisturbanceType",
    "PeatlandStatus",
    "ManagementPractice",
    "InputLevel",
    "ComplianceStatus",
    "ReportingPeriod",
    # Constant tables
    "GWP_VALUES",
    "IPCC_AGB_DEFAULTS",
    "ROOT_SHOOT_RATIOS",
    "DEAD_WOOD_FRACTION",
    "LITTER_STOCKS",
    "SOC_REFERENCE_STOCKS",
    "SOC_LAND_USE_FACTORS",
    "SOC_MANAGEMENT_FACTORS",
    "SOC_INPUT_FACTORS",
    "BIOMASS_GROWTH_RATES",
    "CARBON_FRACTION",
    "COMBUSTION_FACTORS",
    "FIRE_EMISSION_FACTORS",
    "PEATLAND_EF",
    "N2O_SOIL_EF",
    "CONVERSION_FACTOR_CO2_C",
    # Data models
    "LandParcelInfo",
    "CarbonStockSnapshot",
    "EmissionFactorRecord",
    "LandUseTransitionRecord",
    "CalculationRequest",
    "CalculationResult",
    "CalculationDetailResult",
    "BatchCalculationRequest",
    "BatchCalculationResult",
    "SOCAssessmentRequest",
    "SOCAssessmentResult",
    "UncertaintyRequest",
    "UncertaintyResult",
    "ComplianceCheckResult",
    "AggregationRequest",
    "AggregationResult",
    # Constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_POOLS_PER_CALC",
    "MAX_PARCELS_PER_TENANT",
    "DEFAULT_TRANSITION_YEARS",
]
