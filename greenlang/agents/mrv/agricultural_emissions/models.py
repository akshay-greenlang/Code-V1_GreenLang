# -*- coding: utf-8 -*-
"""
Agricultural Emissions Agent Data Models - AGENT-MRV-008

Pydantic v2 data models for the Agricultural Emissions Agent SDK
covering GHG Protocol Scope 1 agricultural emission calculations
including:
- 20 IPCC animal types for enteric fermentation (dairy cattle, non-dairy
  cattle, buffalo, sheep, goats, camels, horses, mules/asses, swine
  market, swine breeding, poultry layers, poultry broilers, turkeys,
  ducks, deer, elk, alpacas/llamas, rabbits, fur-bearing, other)
- 15 AWMS (Animal Waste Management System) manure types (pasture/range,
  daily spread, solid storage, dry lot, liquid slurry no crust, liquid
  slurry with crust, uncovered anaerobic lagoon, covered anaerobic
  lagoon, pit storage <1m, pit storage >1m, deep bedding no mix,
  deep bedding active mix, composting static, composting intensive,
  anaerobic digester)
- 12 crop types for agricultural soils N2O (wheat, corn/maize, rice,
  sugarcane, cotton, soybean, barley, oats, sorghum, millet, other
  cereals, pulses)
- 8 fertilizer types (synthetic N, organic manure, compost, sewage
  sludge, crop residue, limestone, dolomite, urea)
- Rice cultivation CH4 with 7 water regimes and 5 organic amendments
- Field burning of crop residues (CH4 and N2O)
- Liming and urea application CO2
- 6 calculation methods (IPCC Tier 1/2/3, mass balance, direct
  measurement, spend-based)
- Monte Carlo uncertainty quantification
- Multi-framework regulatory compliance (IPCC, GHG Protocol,
  CSRD, EPA, FAO GLEAM, SBTi FLAG, ISO 14064)
- SHA-256 provenance chain for complete audit trails

Enumerations (18):
    - AnimalType, ManureSystem, CropType, FertilizerType,
      WaterRegime, OrganicAmendment, CalculationMethod, EmissionGas,
      GWPSource, EmissionFactorSource, DataQualityTier, FarmType,
      ClimateZone, EmissionSource, ComplianceStatus, ReportingPeriod,
      PreSeasonFlooding, SoilType

Constants:
    - GWP_VALUES: IPCC AR4/AR5/AR6/AR6_20YR GWP values (Decimal)
    - ENTERIC_EF_TIER1: Tier 1 enteric EFs by animal x region (Decimal)
    - MANURE_VS_DEFAULTS: Volatile solids excretion by animal (Decimal)
    - MANURE_BO_VALUES: Maximum CH4 producing capacity by animal (Decimal)
    - MANURE_NEX_VALUES: Nitrogen excretion rates by animal (Decimal)
    - SOIL_N2O_EF: Direct and indirect N2O emission factors
    - INDIRECT_N2O_FRACTIONS: Volatilisation and leaching fractions
    - LIMING_EF: Emission factors for limestone and dolomite
    - UREA_EF: Emission factor for urea application
    - RICE_BASELINE_EF: Baseline CH4 EF for rice cultivation
    - RICE_WATER_REGIME_SF: Water regime scaling factors
    - RICE_ORGANIC_CFOA: Organic amendment conversion factors
    - FIELD_BURNING_EF: Field burning EFs by crop type
    - CONVERSION_C_TO_CO2: Carbon to CO2 ratio (44/12)
    - CONVERSION_N_TO_N2O: Nitrogen to N2O ratio (44/28)

Data Models (18):
    - FarmInfo, LivestockPopulation, ManureSystemAllocation,
      FeedCharacteristics, EntericCalculationRequest,
      ManureCalculationRequest, CroplandInput, RiceFieldInput,
      FieldBurningInput, CalculationRequest, GasEmissionDetail,
      CalculationResult, BatchCalculationRequest,
      BatchCalculationResult, ComplianceCheckResult,
      UncertaintyRequest, UncertaintyResult, AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator

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

#: Maximum livestock populations per single calculation request.
MAX_LIVESTOCK_PER_CALC: int = 100

#: Maximum farms per tenant.
MAX_FARMS_PER_TENANT: int = 50_000

#: Default fraction of volatile solids that decomposes.
DEFAULT_VS_FRACTION: Decimal = Decimal("0.65")

#: Default methane conversion factor for manure.
DEFAULT_MCF: Decimal = Decimal("0.10")

#: Default oxidation factor for manure CH4.
DEFAULT_OXIDATION_FACTOR: Decimal = Decimal("0.10")

# =============================================================================
# Enumerations (18)
# =============================================================================

class AnimalType(str, Enum):
    """IPCC animal categories for enteric fermentation and manure management.

    Twenty animal types as defined in IPCC 2006 Guidelines Volume 4
    Chapter 10 (Emissions from Livestock and Manure Management) and
    2019 Refinement. Each livestock population is classified into
    exactly one animal type to determine applicable enteric emission
    factors, volatile solids excretion rates, and nitrogen excretion.

    DAIRY_CATTLE: Mature dairy cows, primarily kept for milk
        production. High feed intake drives high enteric CH4.
    NON_DAIRY_CATTLE: All other cattle including beef cattle,
        heifers, steers, bulls, and calves not in milk production.
    BUFFALO: Water buffalo and swamp buffalo, primarily in South
        and Southeast Asia for draft power and milk.
    SHEEP: All sheep categories including ewes, rams, and lambs
        for meat, wool, and milk production.
    GOATS: All goat categories including does, bucks, and kids
        for meat, milk, and fiber production.
    CAMELS: Dromedary (one-hump) and Bactrian (two-hump) camels
        for transport, milk, and meat in arid regions.
    HORSES: All horses including draft, riding, and racing horses.
    MULES_ASSES: Mules and donkeys/asses used for draft and
        transport in developing regions.
    SWINE_MARKET: Market (growing/finishing) swine raised for
        meat production, typically 25-110 kg live weight.
    SWINE_BREEDING: Breeding swine including sows, boars, and
        gilts maintained for reproduction.
    POULTRY_LAYERS: Egg-laying hens in commercial operations
        or backyard flocks. Minimal enteric fermentation.
    POULTRY_BROILERS: Meat chickens raised in intensive or
        semi-intensive production systems.
    TURKEYS: All turkey categories for meat production.
    DUCKS: All duck categories for meat and egg production.
    DEER: Farmed deer including red deer, fallow deer, and
        elk raised for venison and velvet.
    ELK: Farmed elk (wapiti) raised for meat and velvet
        antler production.
    ALPACAS_LLAMAS: South American camelids including alpacas
        (fiber) and llamas (pack/fiber) in pastoral systems.
    RABBITS: Farmed rabbits for meat and fur production.
    FUR_BEARING: Fur-bearing animals (mink, fox, chinchilla)
        raised in captive production systems.
    OTHER_LIVESTOCK: Any livestock not classified above,
        requiring custom emission factors.
    """

    DAIRY_CATTLE = "dairy_cattle"
    NON_DAIRY_CATTLE = "non_dairy_cattle"
    BUFFALO = "buffalo"
    SHEEP = "sheep"
    GOATS = "goats"
    CAMELS = "camels"
    HORSES = "horses"
    MULES_ASSES = "mules_asses"
    SWINE_MARKET = "swine_market"
    SWINE_BREEDING = "swine_breeding"
    POULTRY_LAYERS = "poultry_layers"
    POULTRY_BROILERS = "poultry_broilers"
    TURKEYS = "turkeys"
    DUCKS = "ducks"
    DEER = "deer"
    ELK = "elk"
    ALPACAS_LLAMAS = "alpacas_llamas"
    RABBITS = "rabbits"
    FUR_BEARING = "fur_bearing"
    OTHER_LIVESTOCK = "other_livestock"

class ManureSystem(str, Enum):
    """IPCC Animal Waste Management System (AWMS) types.

    Fifteen manure management system types as defined in IPCC 2006
    Guidelines Volume 4 Chapter 10 Table 10.17. The AWMS type
    determines the methane conversion factor (MCF) and nitrogen
    loss pathways for manure emission calculations.

    PASTURE_RANGE_PADDOCK: Manure deposited on pasture, range, or
        paddock by grazing animals. MCF = 0.01-0.02.
    DAILY_SPREAD: Manure collected and spread on fields daily
        or within 24 hours. MCF = 0.001-0.005.
    SOLID_STORAGE: Manure stored in bulk as solid (>20% DM) in
        open or covered piles. MCF = 0.02-0.05.
    DRY_LOT: Manure deposited on unpaved open lots where animals
        are confined. MCF = 0.01-0.05.
    LIQUID_SLURRY_NO_CRUST: Liquid manure (<15% DM) stored in
        tanks or pits without natural crust. MCF = 0.10-0.80.
    LIQUID_SLURRY_WITH_CRUST: Liquid manure stored with natural
        crust formation reducing CH4 emissions. MCF = 0.10-0.40.
    UNCOVERED_ANAEROBIC_LAGOON: Open anaerobic lagoon for liquid
        manure treatment. High MCF = 0.66-0.80.
    COVERED_ANAEROBIC_LAGOON: Anaerobic lagoon with impermeable
        cover and biogas collection. MCF = 0.66-0.80 (recovered).
    PIT_STORAGE_BELOW_1M: Slurry stored in shallow pits (<1m)
        below animal confinement areas. MCF = 0.03-0.30.
    PIT_STORAGE_ABOVE_1M: Slurry stored in deep pits (>1m)
        below animal confinement areas. MCF = 0.03-0.30.
    DEEP_BEDDING_NO_MIX: Deep litter systems where manure
        accumulates without regular mixing. MCF = 0.10-0.40.
    DEEP_BEDDING_ACTIVE_MIX: Deep litter systems with active
        mixing or turning of bedding material. MCF = 0.17-0.44.
    COMPOSTING_STATIC: Static pile or passive aeration composting
        of manure. MCF = 0.005-0.01.
    COMPOSTING_INTENSIVE: Intensive composting with forced aeration
        or frequent turning. MCF = 0.005-0.01.
    ANAEROBIC_DIGESTER: Enclosed anaerobic digester producing
        biogas for energy recovery. MCF = 0.00-0.10.
    """

    PASTURE_RANGE_PADDOCK = "pasture_range_paddock"
    DAILY_SPREAD = "daily_spread"
    SOLID_STORAGE = "solid_storage"
    DRY_LOT = "dry_lot"
    LIQUID_SLURRY_NO_CRUST = "liquid_slurry_no_crust"
    LIQUID_SLURRY_WITH_CRUST = "liquid_slurry_with_crust"
    UNCOVERED_ANAEROBIC_LAGOON = "uncovered_anaerobic_lagoon"
    COVERED_ANAEROBIC_LAGOON = "covered_anaerobic_lagoon"
    PIT_STORAGE_BELOW_1M = "pit_storage_below_1m"
    PIT_STORAGE_ABOVE_1M = "pit_storage_above_1m"
    DEEP_BEDDING_NO_MIX = "deep_bedding_no_mix"
    DEEP_BEDDING_ACTIVE_MIX = "deep_bedding_active_mix"
    COMPOSTING_STATIC = "composting_static"
    COMPOSTING_INTENSIVE = "composting_intensive"
    ANAEROBIC_DIGESTER = "anaerobic_digester"

class CropType(str, Enum):
    """Crop types for agricultural soil N2O and field burning calculations.

    Twelve crop types covering major global agricultural commodities.
    Crop type determines residue-to-product ratios, dry matter content,
    nitrogen content in residues, and field burning emission factors.

    WHEAT: All wheat varieties (bread, durum, spelt). Residue = straw.
    CORN_MAIZE: Maize/corn for grain or silage. Residue = stover.
    RICE: Paddy rice (irrigated and rainfed). Residue = straw.
    SUGARCANE: Sugarcane for sugar and ethanol. Residue = trash/tops.
    COTTON: Cotton for lint and seed. Residue = stalks.
    SOYBEAN: Soybean for oilseed/meal. N-fixing legume.
    BARLEY: Barley for malt and feed. Residue = straw.
    OATS: Oats for food and feed. Residue = straw.
    SORGHUM: Grain sorghum for food and feed. Residue = stover.
    MILLET: All millet types (pearl, finger, foxtail). Residue = straw.
    OTHER_CEREALS: Other cereal crops (rye, triticale, buckwheat).
    PULSES: Leguminous crops (lentils, chickpeas, beans, peas).
        N-fixing crops with lower fertilizer N requirements.
    """

    WHEAT = "wheat"
    CORN_MAIZE = "corn_maize"
    RICE = "rice"
    SUGARCANE = "sugarcane"
    COTTON = "cotton"
    SOYBEAN = "soybean"
    BARLEY = "barley"
    OATS = "oats"
    SORGHUM = "sorghum"
    MILLET = "millet"
    OTHER_CEREALS = "other_cereals"
    PULSES = "pulses"

class FertilizerType(str, Enum):
    """Fertilizer and amendment types for agricultural soil N2O calculations.

    Eight fertilizer and amendment types covering synthetic nitrogen,
    organic nitrogen sources, and liming materials. The fertilizer
    type determines direct and indirect N2O emission factors and
    volatilisation fractions per IPCC 2006 Volume 4 Chapter 11.

    SYNTHETIC_N: Manufactured nitrogen fertilizers (urea, ammonium
        nitrate, anhydrous ammonia, UAN solutions, CAN, MAP, DAP).
    ORGANIC_MANURE: Raw or treated animal manure applied to soils.
        N content varies by animal type and management.
    COMPOST: Composted organic materials including manure-based
        and green waste compost. Lower available N.
    SEWAGE_SLUDGE: Biosolids from municipal wastewater treatment
        applied as soil amendment. Contains organic N.
    CROP_RESIDUE: Crop residues returned to soil including
        above-ground residues, below-ground residues, and roots.
    LIMESTONE: Agricultural limestone (CaCO3) applied for soil
        pH amendment. Source of CO2 emissions.
    DOLOMITE: Dolomitic limestone (CaMg(CO3)2) applied for soil
        pH and magnesium amendment. Higher CO2 EF than limestone.
    UREA: Urea (CO(NH2)2) applied as nitrogen fertilizer. Releases
        CO2 upon hydrolysis in addition to N2O from nitrogen.
    """

    SYNTHETIC_N = "synthetic_n"
    ORGANIC_MANURE = "organic_manure"
    COMPOST = "compost"
    SEWAGE_SLUDGE = "sewage_sludge"
    CROP_RESIDUE = "crop_residue"
    LIMESTONE = "limestone"
    DOLOMITE = "dolomite"
    UREA = "urea"

class WaterRegime(str, Enum):
    """Rice paddy water management regimes for CH4 emission scaling.

    Seven water regime types per IPCC 2006 Guidelines Volume 4
    Chapter 5. Water management is the primary determinant of rice
    paddy CH4 emissions, with continuously flooded fields producing
    the highest emissions and upland rice the lowest.

    CONTINUOUSLY_FLOODED: Paddy fields flooded throughout the
        entire growing season without drainage. Scaling factor = 1.0.
    INTERMITTENT_SINGLE: Single drainage period during the
        growing season. Scaling factor = 0.60.
    INTERMITTENT_MULTIPLE: Multiple aeration/drainage periods
        during the growing season. Scaling factor = 0.52.
    RAINFED_REGULAR: Rainfed paddy with regular rainfall
        maintaining shallow flooding. Scaling factor = 0.28.
    RAINFED_DROUGHT: Rainfed paddy prone to drought periods
        with intermittent dry spells. Scaling factor = 0.25.
    DEEP_WATER: Deep water rice (>50 cm water depth) grown in
        flood-prone areas. Scaling factor = 0.31.
    UPLAND: Upland rice grown in non-flooded aerobic conditions.
        Minimal CH4 emission. Scaling factor = 0.0.
    """

    CONTINUOUSLY_FLOODED = "continuously_flooded"
    INTERMITTENT_SINGLE = "intermittent_single"
    INTERMITTENT_MULTIPLE = "intermittent_multiple"
    RAINFED_REGULAR = "rainfed_regular"
    RAINFED_DROUGHT = "rainfed_drought"
    DEEP_WATER = "deep_water"
    UPLAND = "upland"

class OrganicAmendment(str, Enum):
    """Organic amendment types for rice paddy CH4 emission scaling.

    Five organic amendment types per IPCC 2006 Guidelines Volume 4
    Chapter 5 Table 5.14. Organic amendments increase CH4 emissions
    from rice paddies by providing additional substrate for
    methanogenic bacteria. The conversion factor (CFOA) scales
    the baseline emission factor.

    STRAW_SHORT: Rice straw incorporated <30 days before
        cultivation. Highest CH4 enhancement. CFOA = 1.0.
    STRAW_LONG: Rice straw incorporated >30 days before
        cultivation. Lower enhancement. CFOA = 0.29.
    COMPOST: Composted organic materials applied to paddy.
        Moderate enhancement. CFOA = 0.05.
    FARM_YARD_MANURE: Raw or partially decomposed farmyard
        manure applied to paddy. CFOA = 0.14.
    GREEN_MANURE: Green manure crops incorporated into paddy
        soil before flooding. CFOA = 0.50.
    """

    STRAW_SHORT = "straw_short"
    STRAW_LONG = "straw_long"
    COMPOST = "compost"
    FARM_YARD_MANURE = "farm_yard_manure"
    GREEN_MANURE = "green_manure"

class CalculationMethod(str, Enum):
    """Methodology for calculating agricultural emissions.

    Six calculation methods spanning IPCC Tier 1 through Tier 3
    and supplementary approaches. The method determines required
    input data granularity and expected uncertainty range.

    IPCC_TIER_1: IPCC Tier 1 using global default emission factors.
        Simplest approach with highest uncertainty. Requires only
        animal population or fertilizer mass data. Uncertainty +/-50%.
    IPCC_TIER_2: IPCC Tier 2 using country-specific emission factors
        and enhanced activity data (feed intake, VS, Bo, Nex).
        Moderate uncertainty +/-25%.
    IPCC_TIER_3: IPCC Tier 3 using mechanistic models, farm-specific
        measurements, and detailed process data. Lowest uncertainty.
    MASS_BALANCE: Nutrient mass balance approach tracking nitrogen
        inputs, outputs, and transformations through the farming
        system. Used for system-level verification.
    DIRECT_MEASUREMENT: Direct GHG flux measurement using eddy
        covariance, respiration chambers, or portable analyzers.
    SPEND_BASED: Economic allocation using agricultural expenditure
        data with sector-specific emission intensity factors.
    """

    IPCC_TIER_1 = "ipcc_tier_1"
    IPCC_TIER_2 = "ipcc_tier_2"
    IPCC_TIER_3 = "ipcc_tier_3"
    MASS_BALANCE = "mass_balance"
    DIRECT_MEASUREMENT = "direct_measurement"
    SPEND_BASED = "spend_based"

class EmissionGas(str, Enum):
    """Greenhouse gases tracked in agricultural emission calculations.

    CO2: Carbon dioxide - from liming, urea application, and fuel
        combustion on farms. Not from enteric or manure sources.
    CH4: Methane - from enteric fermentation, manure management,
        and rice cultivation. Primary agricultural GHG.
    N2O: Nitrous oxide - from agricultural soils (direct and
        indirect), manure management, and field burning.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"

class GWPSource(str, Enum):
    """IPCC Assessment Report source for Global Warming Potential values.

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
    """Authoritative source for agricultural emission factors.

    IPCC_2006: IPCC 2006 Guidelines for National Greenhouse Gas
        Inventories, Volume 4 (Agriculture, Forestry, and Other
        Land Use - AFOLU).
    IPCC_2019: 2019 Refinement to the 2006 IPCC Guidelines,
        updated enteric and manure EFs.
    EPA_AP42: US EPA AP-42 Compilation of Air Pollutant Emission
        Factors for agricultural operations.
    DEFRA: UK DEFRA/BEIS Greenhouse Gas Conversion Factors
        (updated annually for agriculture sector).
    ECOINVENT: Ecoinvent database life cycle emission factors
        for agricultural products and processes.
    NATIONAL_INVENTORY: Country-specific national inventory
        emission factors (Tier 2 data from NIR submissions).
    CUSTOM: User-provided custom emission factors from farm
        measurements or site-specific studies (Tier 3).
    """

    IPCC_2006 = "IPCC_2006"
    IPCC_2019 = "IPCC_2019"
    EPA_AP42 = "EPA_AP42"
    DEFRA = "DEFRA"
    ECOINVENT = "ECOINVENT"
    NATIONAL_INVENTORY = "NATIONAL_INVENTORY"
    CUSTOM = "CUSTOM"

class DataQualityTier(str, Enum):
    """IPCC data quality tier for input data and emission factors.

    TIER_1: Global default values from IPCC Guidelines. Highest
        uncertainty, lowest data requirements.
    TIER_2: Country-specific or regional data. Moderate uncertainty,
        requires national statistics or regional surveys.
    TIER_3: Farm-specific measured data. Lowest uncertainty,
        requires direct measurements or detailed records.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"

class FarmType(str, Enum):
    """Type of agricultural operation.

    Determines applicable emission source categories, regulatory
    requirements, and default emission factor sets.

    DAIRY_FARM: Dairy cattle farm primarily producing milk.
        Enteric + manure dominant emission sources.
    BEEF_RANCH: Beef cattle ranch for meat production.
        Enteric fermentation dominant.
    MIXED_LIVESTOCK: Farm with multiple livestock species.
    CROP_FARM: Arable crop farm without significant livestock.
        Agricultural soils N2O dominant.
    RICE_FARM: Rice paddy farm. Rice cultivation CH4 dominant.
    MIXED_CROP_LIVESTOCK: Integrated crop-livestock farming system.
    POULTRY_FARM: Poultry production (layers or broilers).
        Manure management dominant.
    OTHER: Any farm type not classified above.
    """

    DAIRY_FARM = "dairy_farm"
    BEEF_RANCH = "beef_ranch"
    MIXED_LIVESTOCK = "mixed_livestock"
    CROP_FARM = "crop_farm"
    RICE_FARM = "rice_farm"
    MIXED_CROP_LIVESTOCK = "mixed_crop_livestock"
    POULTRY_FARM = "poultry_farm"
    OTHER = "other"

class ClimateZone(str, Enum):
    """IPCC climate zones for agricultural emission factor stratification.

    Eight climate zones as defined in IPCC 2006 Guidelines Volume 4
    Annex 3A.5. Climate zone determines manure MCF values, soil N2O
    emission factors, rice baseline EFs, and enteric EF adjustments.

    TROPICAL_WET: Tropical wet climate. Mean annual temp > 20C,
        MAP > 1000 mm. High decomposition and emission rates.
    TROPICAL_DRY: Tropical dry/semi-arid. Mean annual temp > 20C,
        MAP < 1000 mm.
    WARM_TEMPERATE_WET: Warm temperate moist. Mean annual temp
        10-20C, MAP > 500 mm.
    WARM_TEMPERATE_DRY: Warm temperate dry. Mean annual temp
        10-20C, MAP < 500 mm.
    COOL_TEMPERATE_WET: Cool temperate moist. Mean annual temp
        0-10C, MAP > 500 mm.
    COOL_TEMPERATE_DRY: Cool temperate dry. Mean annual temp
        0-10C, MAP < 500 mm.
    BOREAL_WET: Boreal moist. Mean annual temp < 0C,
        MAP > 400 mm.
    BOREAL_DRY: Boreal dry. Mean annual temp < 0C,
        MAP < 400 mm.
    """

    TROPICAL_WET = "tropical_wet"
    TROPICAL_DRY = "tropical_dry"
    WARM_TEMPERATE_WET = "warm_temperate_wet"
    WARM_TEMPERATE_DRY = "warm_temperate_dry"
    COOL_TEMPERATE_WET = "cool_temperate_wet"
    COOL_TEMPERATE_DRY = "cool_temperate_dry"
    BOREAL_WET = "boreal_wet"
    BOREAL_DRY = "boreal_dry"

class EmissionSource(str, Enum):
    """Agricultural emission source categories.

    Six emission source categories covering all agricultural GHG
    sources as defined in IPCC 2006 Guidelines Volume 4.

    ENTERIC_FERMENTATION: CH4 from microbial fermentation in the
        digestive systems of ruminant and non-ruminant livestock.
    MANURE_MANAGEMENT: CH4 and N2O from storage, treatment, and
        utilization of animal manure and urine.
    AGRICULTURAL_SOILS: Direct and indirect N2O from managed
        agricultural soils receiving nitrogen inputs.
    RICE_CULTIVATION: CH4 from anaerobic decomposition of organic
        matter in flooded rice paddy fields.
    LIMING_UREA: CO2 from application of limestone, dolomite,
        and urea to agricultural soils.
    FIELD_BURNING: CH4 and N2O from burning of crop residues
        in agricultural fields.
    """

    ENTERIC_FERMENTATION = "enteric_fermentation"
    MANURE_MANAGEMENT = "manure_management"
    AGRICULTURAL_SOILS = "agricultural_soils"
    RICE_CULTIVATION = "rice_cultivation"
    LIMING_UREA = "liming_urea"
    FIELD_BURNING = "field_burning"

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

class PreSeasonFlooding(str, Enum):
    """Pre-season water status for rice paddy CH4 scaling.

    Pre-season flooding history affects the baseline CH4 emission
    factor for rice paddies per IPCC 2006 Vol 4 Table 5.12.

    NOT_FLOODED_SHORT: Not flooded for <180 days before cultivation.
        Scaling factor SFp = 1.0.
    FLOODED_LONG: Flooded for >30 days before cultivation period.
        Higher organic matter decomposition. SFp = 1.22.
    NOT_KNOWN: Pre-season status not known. Default SFp = 1.0.
    """

    NOT_FLOODED_SHORT = "not_flooded_short"
    FLOODED_LONG = "flooded_long"
    NOT_KNOWN = "not_known"

class SoilType(str, Enum):
    """Soil types affecting N2O emission factors.

    Five soil types relevant to N2O emission rates from
    agricultural soils. Organic/histosol soils have higher
    baseline N2O emissions due to nitrogen mineralization.

    MINERAL: Standard mineral soils (most agricultural land).
    ORGANIC_HISTOSOL: Organic/histosol soils (peat soils) with
        >20% organic carbon. Higher N2O from mineralization.
    SANDY: Sandy soils with high drainage. Lower denitrification
        but higher leaching losses.
    CLAY: Clay soils with poor drainage. Higher denitrification
        and N2O production potential.
    PEAT: Drained peatland used for agriculture. Very high N2O
        from organic matter decomposition.

from greenlang.schemas import GreenLangBase, utcnow
    """

    MINERAL = "mineral"
    ORGANIC_HISTOSOL = "organic_histosol"
    SANDY = "sandy"
    CLAY = "clay"
    PEAT = "peat"

# =============================================================================
# Constant Tables (all Decimal for deterministic arithmetic)
# =============================================================================

# ---------------------------------------------------------------------------
# GWP values by IPCC Assessment Report
# Agricultural CH4 is biogenic; AR6 distinguishes fossil vs biogenic.
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[GWPSource, Dict[str, Decimal]] = {
    GWPSource.AR4: {
        "CO2": Decimal("1"),
        "CH4_fossil": Decimal("25"),
        "CH4_biogenic": Decimal("25"),
        "N2O": Decimal("298"),
    },
    GWPSource.AR5: {
        "CO2": Decimal("1"),
        "CH4_fossil": Decimal("28"),
        "CH4_biogenic": Decimal("28"),
        "N2O": Decimal("265"),
    },
    GWPSource.AR6: {
        "CO2": Decimal("1"),
        "CH4_fossil": Decimal("29.8"),
        "CH4_biogenic": Decimal("27.0"),
        "N2O": Decimal("273"),
    },
    GWPSource.AR6_20YR: {
        "CO2": Decimal("1"),
        "CH4_fossil": Decimal("82.5"),
        "CH4_biogenic": Decimal("80.8"),
        "N2O": Decimal("273"),
    },
}

# ---------------------------------------------------------------------------
# Molecular weight / stoichiometric conversion factors
# ---------------------------------------------------------------------------

#: Carbon to CO2 molecular weight ratio (44/12).
CONVERSION_C_TO_CO2: Decimal = Decimal("3.66667")

#: Nitrogen to N2O molecular weight ratio (44/28).
CONVERSION_N_TO_N2O: Decimal = Decimal("1.57143")

#: CH4 to C molecular weight ratio (16/12).
CH4_C_RATIO: Decimal = Decimal("1.33333")

# ---------------------------------------------------------------------------
# Tier 1 Enteric Fermentation Emission Factors
# IPCC 2006 Guidelines Volume 4 Table 10.10 and 10.11
# Units: kg CH4 per head per year
# Keyed by (AnimalType, region) where region is "developed"/"developing"
# ---------------------------------------------------------------------------

ENTERIC_EF_TIER1: Dict[AnimalType, Dict[str, Decimal]] = {
    AnimalType.DAIRY_CATTLE: {
        "developed": Decimal("128"),
        "developing": Decimal("89"),
    },
    AnimalType.NON_DAIRY_CATTLE: {
        "developed": Decimal("53"),
        "developing": Decimal("44"),
    },
    AnimalType.BUFFALO: {
        "developed": Decimal("55"),
        "developing": Decimal("55"),
    },
    AnimalType.SHEEP: {
        "developed": Decimal("8"),
        "developing": Decimal("5"),
    },
    AnimalType.GOATS: {
        "developed": Decimal("5"),
        "developing": Decimal("5"),
    },
    AnimalType.CAMELS: {
        "developed": Decimal("46"),
        "developing": Decimal("46"),
    },
    AnimalType.HORSES: {
        "developed": Decimal("18"),
        "developing": Decimal("18"),
    },
    AnimalType.MULES_ASSES: {
        "developed": Decimal("10"),
        "developing": Decimal("10"),
    },
    AnimalType.SWINE_MARKET: {
        "developed": Decimal("1.5"),
        "developing": Decimal("1.0"),
    },
    AnimalType.SWINE_BREEDING: {
        "developed": Decimal("1.5"),
        "developing": Decimal("1.0"),
    },
    AnimalType.POULTRY_LAYERS: {
        "developed": Decimal("0.0"),
        "developing": Decimal("0.0"),
    },
    AnimalType.POULTRY_BROILERS: {
        "developed": Decimal("0.0"),
        "developing": Decimal("0.0"),
    },
    AnimalType.TURKEYS: {
        "developed": Decimal("0.0"),
        "developing": Decimal("0.0"),
    },
    AnimalType.DUCKS: {
        "developed": Decimal("0.0"),
        "developing": Decimal("0.0"),
    },
    AnimalType.DEER: {
        "developed": Decimal("20"),
        "developing": Decimal("20"),
    },
    AnimalType.ELK: {
        "developed": Decimal("25"),
        "developing": Decimal("25"),
    },
    AnimalType.ALPACAS_LLAMAS: {
        "developed": Decimal("8"),
        "developing": Decimal("8"),
    },
    AnimalType.RABBITS: {
        "developed": Decimal("0.2"),
        "developing": Decimal("0.2"),
    },
    AnimalType.FUR_BEARING: {
        "developed": Decimal("0.1"),
        "developing": Decimal("0.1"),
    },
    AnimalType.OTHER_LIVESTOCK: {
        "developed": Decimal("10"),
        "developing": Decimal("10"),
    },
}

# ---------------------------------------------------------------------------
# Volatile Solids (VS) excretion defaults
# IPCC 2006 Volume 4 Table 10.13A
# Units: kg VS per head per day
# ---------------------------------------------------------------------------

MANURE_VS_DEFAULTS: Dict[AnimalType, Decimal] = {
    AnimalType.DAIRY_CATTLE: Decimal("5.40"),
    AnimalType.NON_DAIRY_CATTLE: Decimal("3.64"),
    AnimalType.BUFFALO: Decimal("3.90"),
    AnimalType.SHEEP: Decimal("0.32"),
    AnimalType.GOATS: Decimal("0.35"),
    AnimalType.CAMELS: Decimal("2.40"),
    AnimalType.HORSES: Decimal("2.10"),
    AnimalType.MULES_ASSES: Decimal("1.20"),
    AnimalType.SWINE_MARKET: Decimal("0.50"),
    AnimalType.SWINE_BREEDING: Decimal("0.55"),
    AnimalType.POULTRY_LAYERS: Decimal("0.02"),
    AnimalType.POULTRY_BROILERS: Decimal("0.01"),
    AnimalType.TURKEYS: Decimal("0.04"),
    AnimalType.DUCKS: Decimal("0.02"),
    AnimalType.DEER: Decimal("0.90"),
    AnimalType.ELK: Decimal("1.30"),
    AnimalType.ALPACAS_LLAMAS: Decimal("0.72"),
    AnimalType.RABBITS: Decimal("0.04"),
    AnimalType.FUR_BEARING: Decimal("0.03"),
    AnimalType.OTHER_LIVESTOCK: Decimal("1.00"),
}

# ---------------------------------------------------------------------------
# Maximum CH4 producing capacity (Bo)
# IPCC 2006 Volume 4 Table 10.16
# Units: m3 CH4 per kg VS
# ---------------------------------------------------------------------------

MANURE_BO_VALUES: Dict[AnimalType, Decimal] = {
    AnimalType.DAIRY_CATTLE: Decimal("0.24"),
    AnimalType.NON_DAIRY_CATTLE: Decimal("0.18"),
    AnimalType.BUFFALO: Decimal("0.10"),
    AnimalType.SHEEP: Decimal("0.19"),
    AnimalType.GOATS: Decimal("0.18"),
    AnimalType.CAMELS: Decimal("0.15"),
    AnimalType.HORSES: Decimal("0.33"),
    AnimalType.MULES_ASSES: Decimal("0.33"),
    AnimalType.SWINE_MARKET: Decimal("0.45"),
    AnimalType.SWINE_BREEDING: Decimal("0.45"),
    AnimalType.POULTRY_LAYERS: Decimal("0.39"),
    AnimalType.POULTRY_BROILERS: Decimal("0.36"),
    AnimalType.TURKEYS: Decimal("0.36"),
    AnimalType.DUCKS: Decimal("0.36"),
    AnimalType.DEER: Decimal("0.18"),
    AnimalType.ELK: Decimal("0.18"),
    AnimalType.ALPACAS_LLAMAS: Decimal("0.19"),
    AnimalType.RABBITS: Decimal("0.32"),
    AnimalType.FUR_BEARING: Decimal("0.28"),
    AnimalType.OTHER_LIVESTOCK: Decimal("0.17"),
}

# ---------------------------------------------------------------------------
# Nitrogen excretion rates (Nex)
# IPCC 2006 Volume 4 Table 10.19
# Units: kg N per head per year
# ---------------------------------------------------------------------------

MANURE_NEX_VALUES: Dict[AnimalType, Decimal] = {
    AnimalType.DAIRY_CATTLE: Decimal("100.0"),
    AnimalType.NON_DAIRY_CATTLE: Decimal("40.0"),
    AnimalType.BUFFALO: Decimal("40.0"),
    AnimalType.SHEEP: Decimal("12.0"),
    AnimalType.GOATS: Decimal("12.0"),
    AnimalType.CAMELS: Decimal("36.0"),
    AnimalType.HORSES: Decimal("26.0"),
    AnimalType.MULES_ASSES: Decimal("18.0"),
    AnimalType.SWINE_MARKET: Decimal("7.0"),
    AnimalType.SWINE_BREEDING: Decimal("14.0"),
    AnimalType.POULTRY_LAYERS: Decimal("0.6"),
    AnimalType.POULTRY_BROILERS: Decimal("0.3"),
    AnimalType.TURKEYS: Decimal("0.7"),
    AnimalType.DUCKS: Decimal("0.4"),
    AnimalType.DEER: Decimal("14.0"),
    AnimalType.ELK: Decimal("20.0"),
    AnimalType.ALPACAS_LLAMAS: Decimal("10.0"),
    AnimalType.RABBITS: Decimal("1.8"),
    AnimalType.FUR_BEARING: Decimal("1.4"),
    AnimalType.OTHER_LIVESTOCK: Decimal("12.0"),
}

# ---------------------------------------------------------------------------
# Manure Management MCF values by system and climate
# IPCC 2006 Volume 4 Table 10.17
# Keyed by (ManureSystem, temperature_class) where temperature_class
# is "cool" (<15C), "temperate" (15-25C), or "warm" (>25C)
# ---------------------------------------------------------------------------

MANURE_MCF_VALUES: Dict[ManureSystem, Dict[str, Decimal]] = {
    ManureSystem.PASTURE_RANGE_PADDOCK: {
        "cool": Decimal("0.01"),
        "temperate": Decimal("0.015"),
        "warm": Decimal("0.02"),
    },
    ManureSystem.DAILY_SPREAD: {
        "cool": Decimal("0.001"),
        "temperate": Decimal("0.005"),
        "warm": Decimal("0.005"),
    },
    ManureSystem.SOLID_STORAGE: {
        "cool": Decimal("0.02"),
        "temperate": Decimal("0.04"),
        "warm": Decimal("0.05"),
    },
    ManureSystem.DRY_LOT: {
        "cool": Decimal("0.01"),
        "temperate": Decimal("0.015"),
        "warm": Decimal("0.05"),
    },
    ManureSystem.LIQUID_SLURRY_NO_CRUST: {
        "cool": Decimal("0.10"),
        "temperate": Decimal("0.35"),
        "warm": Decimal("0.80"),
    },
    ManureSystem.LIQUID_SLURRY_WITH_CRUST: {
        "cool": Decimal("0.10"),
        "temperate": Decimal("0.17"),
        "warm": Decimal("0.40"),
    },
    ManureSystem.UNCOVERED_ANAEROBIC_LAGOON: {
        "cool": Decimal("0.66"),
        "temperate": Decimal("0.74"),
        "warm": Decimal("0.80"),
    },
    ManureSystem.COVERED_ANAEROBIC_LAGOON: {
        "cool": Decimal("0.66"),
        "temperate": Decimal("0.74"),
        "warm": Decimal("0.80"),
    },
    ManureSystem.PIT_STORAGE_BELOW_1M: {
        "cool": Decimal("0.03"),
        "temperate": Decimal("0.17"),
        "warm": Decimal("0.30"),
    },
    ManureSystem.PIT_STORAGE_ABOVE_1M: {
        "cool": Decimal("0.03"),
        "temperate": Decimal("0.17"),
        "warm": Decimal("0.30"),
    },
    ManureSystem.DEEP_BEDDING_NO_MIX: {
        "cool": Decimal("0.10"),
        "temperate": Decimal("0.17"),
        "warm": Decimal("0.40"),
    },
    ManureSystem.DEEP_BEDDING_ACTIVE_MIX: {
        "cool": Decimal("0.17"),
        "temperate": Decimal("0.30"),
        "warm": Decimal("0.44"),
    },
    ManureSystem.COMPOSTING_STATIC: {
        "cool": Decimal("0.005"),
        "temperate": Decimal("0.005"),
        "warm": Decimal("0.01"),
    },
    ManureSystem.COMPOSTING_INTENSIVE: {
        "cool": Decimal("0.005"),
        "temperate": Decimal("0.005"),
        "warm": Decimal("0.01"),
    },
    ManureSystem.ANAEROBIC_DIGESTER: {
        "cool": Decimal("0.00"),
        "temperate": Decimal("0.01"),
        "warm": Decimal("0.10"),
    },
}

# ---------------------------------------------------------------------------
# Soil N2O emission factors
# IPCC 2006 Volume 4 Chapter 11 Table 11.1 and 11.3
# ---------------------------------------------------------------------------

SOIL_N2O_EF: Dict[str, Decimal] = {
    # EF1: Direct N2O EF from N inputs (kg N2O-N per kg N applied)
    "EF1": Decimal("0.01"),
    # EF2_CG: Direct N2O EF from cultivated organic soils, cool/temperate
    # (kg N2O-N per ha per year)
    "EF2_CG": Decimal("8.0"),
    # EF2_F: Direct N2O EF from cultivated organic soils, tropical
    # (kg N2O-N per ha per year)
    "EF2_F": Decimal("2.5"),
    # EF3_PRP_cattle: EF for N from cattle urine/dung on pasture
    # (kg N2O-N per kg N)
    "EF3_PRP_cattle": Decimal("0.02"),
    # EF3_PRP_other: EF for N from other animal urine/dung on pasture
    # (kg N2O-N per kg N)
    "EF3_PRP_other": Decimal("0.01"),
}

# ---------------------------------------------------------------------------
# Indirect N2O emission factors and fractions
# IPCC 2006 Volume 4 Chapter 11 Tables 11.1, 11.3
# ---------------------------------------------------------------------------

INDIRECT_N2O_FRACTIONS: Dict[str, Decimal] = {
    # Frac_GASF: Fraction of synthetic fertilizer N volatilised as NH3/NOx
    "Frac_GASF": Decimal("0.10"),
    # Frac_GASM: Fraction of organic N (manure) volatilised as NH3/NOx
    "Frac_GASM": Decimal("0.20"),
    # Frac_LEACH: Fraction of N inputs lost via leaching/runoff
    "Frac_LEACH": Decimal("0.30"),
    # EF4: Indirect N2O EF for atmospheric deposition (kg N2O-N per kg NH3-N)
    "EF4": Decimal("0.01"),
    # EF5: Indirect N2O EF for leaching/runoff (kg N2O-N per kg N leached)
    "EF5": Decimal("0.0075"),
}

# ---------------------------------------------------------------------------
# Liming emission factors
# IPCC 2006 Volume 4 Chapter 11 Equation 11.12
# Units: tonnes CO2-C per tonne of material applied
# ---------------------------------------------------------------------------

LIMING_EF: Dict[str, Decimal] = {
    "limestone": Decimal("0.12"),
    "dolomite": Decimal("0.13"),
}

# ---------------------------------------------------------------------------
# Urea emission factor
# IPCC 2006 Volume 4 Chapter 11 Equation 11.13
# Units: tonnes CO2-C per tonne of urea applied
# ---------------------------------------------------------------------------

UREA_EF: Decimal = Decimal("0.20")

# ---------------------------------------------------------------------------
# Rice cultivation CH4 baseline emission factor
# IPCC 2006 Volume 4 Chapter 5 Table 5.11
# Units: kg CH4 per hectare per day (continuously flooded without
# organic amendments, for growing season)
# ---------------------------------------------------------------------------

RICE_BASELINE_EF: Decimal = Decimal("1.30")

# ---------------------------------------------------------------------------
# Rice water regime scaling factors (SFw)
# IPCC 2006 Volume 4 Table 5.12
# Multiply baseline EF by SFw to adjust for water management
# ---------------------------------------------------------------------------

RICE_WATER_REGIME_SF: Dict[WaterRegime, Decimal] = {
    WaterRegime.CONTINUOUSLY_FLOODED: Decimal("1.0"),
    WaterRegime.INTERMITTENT_SINGLE: Decimal("0.60"),
    WaterRegime.INTERMITTENT_MULTIPLE: Decimal("0.52"),
    WaterRegime.RAINFED_REGULAR: Decimal("0.28"),
    WaterRegime.RAINFED_DROUGHT: Decimal("0.25"),
    WaterRegime.DEEP_WATER: Decimal("0.31"),
    WaterRegime.UPLAND: Decimal("0.0"),
}

# ---------------------------------------------------------------------------
# Rice pre-season flooding scaling factors (SFp)
# IPCC 2006 Volume 4 Table 5.13
# ---------------------------------------------------------------------------

RICE_PRESEASON_SF: Dict[PreSeasonFlooding, Decimal] = {
    PreSeasonFlooding.NOT_FLOODED_SHORT: Decimal("1.0"),
    PreSeasonFlooding.FLOODED_LONG: Decimal("1.22"),
    PreSeasonFlooding.NOT_KNOWN: Decimal("1.0"),
}

# ---------------------------------------------------------------------------
# Rice organic amendment conversion factors (CFOA)
# IPCC 2006 Volume 4 Table 5.14
# Used in formula: SFo = (1 + sum(ROAi * CFOAi))^0.59
# where ROAi = application rate in tonnes/ha (dry weight basis)
# ---------------------------------------------------------------------------

RICE_ORGANIC_CFOA: Dict[OrganicAmendment, Decimal] = {
    OrganicAmendment.STRAW_SHORT: Decimal("1.0"),
    OrganicAmendment.STRAW_LONG: Decimal("0.29"),
    OrganicAmendment.COMPOST: Decimal("0.05"),
    OrganicAmendment.FARM_YARD_MANURE: Decimal("0.14"),
    OrganicAmendment.GREEN_MANURE: Decimal("0.50"),
}

# ---------------------------------------------------------------------------
# Field burning emission factors
# IPCC 2006 Volume 4 Chapter 2 Table 2.5 and 2.6
# Keyed by crop type with:
#   RPR: residue-to-product ratio (kg residue / kg product)
#   DM: dry matter fraction of residue
#   CF: combustion factor (fraction of DM burned in field)
#   EF_CH4: emission factor for CH4 (g CH4 per kg dry matter burned)
#   EF_N2O: emission factor for N2O (g N2O per kg dry matter burned)
#   N_content: nitrogen content of residue (fraction of DM)
# ---------------------------------------------------------------------------

FIELD_BURNING_EF: Dict[CropType, Dict[str, Decimal]] = {
    CropType.WHEAT: {
        "RPR": Decimal("1.3"),
        "DM": Decimal("0.88"),
        "CF": Decimal("0.90"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.006"),
    },
    CropType.CORN_MAIZE: {
        "RPR": Decimal("1.0"),
        "DM": Decimal("0.87"),
        "CF": Decimal("0.80"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.006"),
    },
    CropType.RICE: {
        "RPR": Decimal("1.4"),
        "DM": Decimal("0.86"),
        "CF": Decimal("0.80"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.007"),
    },
    CropType.SUGARCANE: {
        "RPR": Decimal("0.30"),
        "DM": Decimal("0.71"),
        "CF": Decimal("0.80"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.004"),
    },
    CropType.COTTON: {
        "RPR": Decimal("2.1"),
        "DM": Decimal("0.91"),
        "CF": Decimal("0.90"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.012"),
    },
    CropType.SOYBEAN: {
        "RPR": Decimal("2.1"),
        "DM": Decimal("0.87"),
        "CF": Decimal("0.90"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.008"),
    },
    CropType.BARLEY: {
        "RPR": Decimal("1.2"),
        "DM": Decimal("0.88"),
        "CF": Decimal("0.90"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.006"),
    },
    CropType.OATS: {
        "RPR": Decimal("1.3"),
        "DM": Decimal("0.87"),
        "CF": Decimal("0.90"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.006"),
    },
    CropType.SORGHUM: {
        "RPR": Decimal("1.4"),
        "DM": Decimal("0.88"),
        "CF": Decimal("0.80"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.006"),
    },
    CropType.MILLET: {
        "RPR": Decimal("1.4"),
        "DM": Decimal("0.86"),
        "CF": Decimal("0.80"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.006"),
    },
    CropType.OTHER_CEREALS: {
        "RPR": Decimal("1.3"),
        "DM": Decimal("0.87"),
        "CF": Decimal("0.85"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.006"),
    },
    CropType.PULSES: {
        "RPR": Decimal("2.1"),
        "DM": Decimal("0.87"),
        "CF": Decimal("0.90"),
        "EF_CH4": Decimal("2.7"),
        "EF_N2O": Decimal("0.07"),
        "N_content": Decimal("0.008"),
    },
}

# ---------------------------------------------------------------------------
# Manure N2O direct emission factor by AWMS
# IPCC 2006 Volume 4 Table 10.21
# kg N2O-N per kg N in manure managed in system
# ---------------------------------------------------------------------------

MANURE_N2O_EF: Dict[ManureSystem, Decimal] = {
    ManureSystem.PASTURE_RANGE_PADDOCK: Decimal("0.0"),
    ManureSystem.DAILY_SPREAD: Decimal("0.0"),
    ManureSystem.SOLID_STORAGE: Decimal("0.005"),
    ManureSystem.DRY_LOT: Decimal("0.02"),
    ManureSystem.LIQUID_SLURRY_NO_CRUST: Decimal("0.0"),
    ManureSystem.LIQUID_SLURRY_WITH_CRUST: Decimal("0.005"),
    ManureSystem.UNCOVERED_ANAEROBIC_LAGOON: Decimal("0.0"),
    ManureSystem.COVERED_ANAEROBIC_LAGOON: Decimal("0.0"),
    ManureSystem.PIT_STORAGE_BELOW_1M: Decimal("0.002"),
    ManureSystem.PIT_STORAGE_ABOVE_1M: Decimal("0.002"),
    ManureSystem.DEEP_BEDDING_NO_MIX: Decimal("0.01"),
    ManureSystem.DEEP_BEDDING_ACTIVE_MIX: Decimal("0.07"),
    ManureSystem.COMPOSTING_STATIC: Decimal("0.006"),
    ManureSystem.COMPOSTING_INTENSIVE: Decimal("0.006"),
    ManureSystem.ANAEROBIC_DIGESTER: Decimal("0.0"),
}

# ---------------------------------------------------------------------------
# CH4 density at STP for volume-to-mass conversion
# Units: kg per m3 at 0 C, 101.325 kPa
# ---------------------------------------------------------------------------

CH4_DENSITY_STP: Decimal = Decimal("0.7168")

# =============================================================================
# Pydantic Data Models (18)
# =============================================================================

class FarmInfo(GreenLangBase):
    """Agricultural farm/operation registration and metadata.

    Represents an agricultural operation with its location, type,
    climate zone, and operational characteristics. Each farm is
    registered under a tenant for multi-tenancy.

    Attributes:
        id: Unique farm identifier (UUID).
        name: Human-readable farm name.
        farm_type: Type of agricultural operation.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        climate_zone: IPCC climate zone for EF stratification.
        country_code: ISO 3166-1 alpha-2 country code.
        region: Developed or developing region classification.
        total_area_ha: Total farm area in hectares.
        cropland_area_ha: Cropland area in hectares.
        pasture_area_ha: Pasture/grassland area in hectares.
        soil_type: Predominant soil type.
        organic_soil_area_ha: Area of organic/histosol soils (ha).
        tenant_id: Owning tenant identifier for multi-tenancy.
        is_active: Whether farm is currently operational.
        created_at: UTC timestamp of farm registration.
        updated_at: UTC timestamp of last update.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique farm identifier (UUID)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable farm name",
    )
    farm_type: FarmType = Field(
        ...,
        description="Type of agricultural operation",
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
        default=ClimateZone.COOL_TEMPERATE_WET,
        description="IPCC climate zone for EF stratification",
    )
    country_code: str = Field(
        default="",
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: str = Field(
        default="developed",
        description="Region classification: 'developed' or 'developing'",
    )
    total_area_ha: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total farm area in hectares",
    )
    cropland_area_ha: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Cropland area in hectares",
    )
    pasture_area_ha: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Pasture/grassland area in hectares",
    )
    soil_type: SoilType = Field(
        default=SoilType.MINERAL,
        description="Predominant soil type",
    )
    organic_soil_area_ha: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Area of organic/histosol soils in hectares",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Owning tenant identifier",
    )
    is_active: bool = Field(
        default=True,
        description="Whether farm is currently operational",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of farm registration",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of last update",
    )

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate region is either 'developed' or 'developing'."""
        v_lower = v.lower().strip()
        if v_lower not in ("developed", "developing"):
            raise ValueError(
                f"region must be 'developed' or 'developing', got '{v}'"
            )
        return v_lower

class LivestockPopulation(GreenLangBase):
    """Livestock population record for a specific animal type.

    Represents the annual average population of a single animal
    type at a farm, with optional Tier 2 parameters for enhanced
    emission calculations.

    Attributes:
        id: Unique record identifier (UUID).
        farm_id: Reference to the farm.
        animal_type: IPCC animal category.
        head_count: Annual average number of animals (head).
        typical_animal_mass_kg: Average live weight per animal (kg).
        region: Developed or developing region classification.
        feed_intake_mj_per_day: Gross energy intake (MJ/head/day)
            for Tier 2 enteric calculations.
        methane_conversion_factor: Ym - fraction of GE converted
            to CH4 (0.0-0.20) for Tier 2.
        vs_rate_override: Override VS excretion rate (kg/head/day).
        bo_override: Override maximum CH4 capacity (m3/kg VS).
        nex_override: Override N excretion rate (kg N/head/yr).
        data_quality_tier: Quality tier of input data.
        reference_year: Year the population data applies to.
        notes: Optional description or source of population data.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID)",
    )
    farm_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the farm",
    )
    animal_type: AnimalType = Field(
        ...,
        description="IPCC animal category",
    )
    head_count: int = Field(
        ...,
        gt=0,
        description="Annual average number of animals (head)",
    )
    typical_animal_mass_kg: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Average live weight per animal (kg)",
    )
    region: str = Field(
        default="developed",
        description="Region classification: 'developed' or 'developing'",
    )
    feed_intake_mj_per_day: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Gross energy intake (MJ/head/day) for Tier 2",
    )
    methane_conversion_factor: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("0.20"),
        description="Ym - fraction of GE converted to CH4 (Tier 2)",
    )
    vs_rate_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Override VS excretion rate (kg/head/day)",
    )
    bo_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Override maximum CH4 capacity (m3/kg VS)",
    )
    nex_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Override N excretion rate (kg N/head/yr)",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Quality tier of input data",
    )
    reference_year: int = Field(
        default=2025,
        ge=1900,
        le=2100,
        description="Year the population data applies to",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional description or source of population data",
    )

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate region is either 'developed' or 'developing'."""
        v_lower = v.lower().strip()
        if v_lower not in ("developed", "developing"):
            raise ValueError(
                f"region must be 'developed' or 'developing', got '{v}'"
            )
        return v_lower

class ManureSystemAllocation(GreenLangBase):
    """Allocation of livestock manure to management systems.

    Represents the fraction of manure from a livestock population
    allocated to a specific Animal Waste Management System (AWMS).
    A livestock population may have multiple allocations summing
    to 1.0 (100%).

    Attributes:
        id: Unique record identifier (UUID).
        livestock_id: Reference to the livestock population.
        manure_system: AWMS type for this allocation.
        allocation_fraction: Fraction of manure allocated (0.0-1.0).
        mcf_override: Override MCF value for this system.
        temperature_class: Temperature classification for MCF lookup.
        ch4_recovery_fraction: Fraction of CH4 recovered (covered
            lagoon, digester). (0.0-1.0).
        notes: Optional notes about the allocation.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID)",
    )
    livestock_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the livestock population",
    )
    manure_system: ManureSystem = Field(
        ...,
        description="AWMS type for this allocation",
    )
    allocation_fraction: Decimal = Field(
        ...,
        gt=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of manure allocated to this system",
    )
    mcf_override: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Override MCF value for this system",
    )
    temperature_class: str = Field(
        default="temperate",
        description="Temperature classification: 'cool', 'temperate', 'warm'",
    )
    ch4_recovery_fraction: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of CH4 recovered from this system",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes about the allocation",
    )

    @field_validator("temperature_class")
    @classmethod
    def validate_temperature_class(cls, v: str) -> str:
        """Validate temperature class is a recognized value."""
        v_lower = v.lower().strip()
        if v_lower not in ("cool", "temperate", "warm"):
            raise ValueError(
                f"temperature_class must be 'cool', 'temperate', or "
                f"'warm', got '{v}'"
            )
        return v_lower

class FeedCharacteristics(GreenLangBase):
    """Feed composition and quality data for Tier 2 enteric calculations.

    Provides detailed feed parameters for enhanced enteric
    fermentation CH4 estimation using IPCC Tier 2 methodology.
    Tier 2 calculates gross energy intake and applies a
    methane conversion factor (Ym).

    Attributes:
        id: Unique record identifier (UUID).
        livestock_id: Reference to the livestock population.
        diet_description: Description of the diet composition.
        crude_protein_pct: Crude protein content (% of DM).
        digestible_energy_pct: Digestible energy as % of GE.
        total_digestible_nutrients_pct: TDN (% of DM).
        neutral_detergent_fiber_pct: NDF (% of DM).
        acid_detergent_fiber_pct: ADF (% of DM).
        ether_extract_pct: Fat content (% of DM).
        ash_pct: Ash content (% of DM).
        dry_matter_intake_kg_per_day: DMI (kg DM/head/day).
        gross_energy_mj_per_kg_dm: GE content (MJ/kg DM).
        forage_fraction: Fraction of diet from forage (0.0-1.0).
        grain_fraction: Fraction of diet from grain (0.0-1.0).
        notes: Optional notes about feed data source.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID)",
    )
    livestock_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the livestock population",
    )
    diet_description: str = Field(
        default="",
        max_length=1000,
        description="Description of the diet composition",
    )
    crude_protein_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("50"),
        description="Crude protein content (% of DM)",
    )
    digestible_energy_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Digestible energy as percentage of GE",
    )
    total_digestible_nutrients_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Total digestible nutrients (% of DM)",
    )
    neutral_detergent_fiber_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Neutral detergent fiber (% of DM)",
    )
    acid_detergent_fiber_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Acid detergent fiber (% of DM)",
    )
    ether_extract_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("20"),
        description="Fat content (% of DM)",
    )
    ash_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("30"),
        description="Ash content (% of DM)",
    )
    dry_matter_intake_kg_per_day: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Dry matter intake (kg DM/head/day)",
    )
    gross_energy_mj_per_kg_dm: Decimal = Field(
        default=Decimal("18.45"),
        gt=Decimal("0"),
        description="Gross energy content (MJ/kg DM)",
    )
    forage_fraction: Decimal = Field(
        default=Decimal("0.50"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of diet from forage",
    )
    grain_fraction: Decimal = Field(
        default=Decimal("0.50"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of diet from grain/concentrate",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes about feed data source",
    )

    @field_validator("grain_fraction")
    @classmethod
    def validate_diet_fractions(cls, v: Decimal, info: Any) -> Decimal:
        """Validate that forage + grain fractions do not exceed 1.0."""
        data = info.data if hasattr(info, "data") else {}
        forage = data.get("forage_fraction", Decimal("0.50"))
        if forage is not None:
            total = v + forage
            if total > Decimal("1.01"):
                raise ValueError(
                    f"forage_fraction ({forage}) + grain_fraction ({v}) "
                    f"must sum to <= 1.0, got {total}"
                )
        return v

class EntericCalculationRequest(GreenLangBase):
    """Request for enteric fermentation CH4 calculation.

    Specifies parameters for calculating methane emissions from
    livestock enteric fermentation. Supports Tier 1 (default EFs)
    and Tier 2 (feed-based GE * Ym) approaches.

    Attributes:
        id: Unique request identifier (UUID).
        livestock: Livestock population for the calculation.
        feed_characteristics: Optional Tier 2 feed parameters.
        calculation_method: Tier 1 or Tier 2 methodology.
        ef_override_kg_ch4_per_head: Override enteric EF
            (kg CH4/head/year) for custom factors.
        gwp_source: GWP source for CO2e conversion.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier (UUID)",
    )
    livestock: LivestockPopulation = Field(
        ...,
        description="Livestock population for the calculation",
    )
    feed_characteristics: Optional[FeedCharacteristics] = Field(
        default=None,
        description="Optional Tier 2 feed parameters",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.IPCC_TIER_1,
        description="Tier 1 or Tier 2 methodology",
    )
    ef_override_kg_ch4_per_head: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Override enteric EF (kg CH4/head/year)",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source for CO2e conversion",
    )

class ManureCalculationRequest(GreenLangBase):
    """Request for manure management CH4 and N2O calculation.

    Specifies parameters for calculating methane and nitrous oxide
    emissions from animal manure management systems. Supports
    multiple AWMS allocations per livestock population.

    Attributes:
        id: Unique request identifier (UUID).
        livestock: Livestock population for the calculation.
        manure_allocations: List of AWMS allocations (must sum
            to 1.0 across all allocations).
        calculation_method: Tier 1 or Tier 2 methodology.
        vs_override_kg_per_day: Override VS excretion rate.
        bo_override_m3_per_kg: Override maximum CH4 capacity.
        nex_override_kg_per_year: Override N excretion rate.
        gwp_source: GWP source for CO2e conversion.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier (UUID)",
    )
    livestock: LivestockPopulation = Field(
        ...,
        description="Livestock population for the calculation",
    )
    manure_allocations: List[ManureSystemAllocation] = Field(
        ...,
        min_length=1,
        description="List of AWMS allocations",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.IPCC_TIER_1,
        description="Tier 1 or Tier 2 methodology",
    )
    vs_override_kg_per_day: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Override VS excretion rate (kg/head/day)",
    )
    bo_override_m3_per_kg: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Override maximum CH4 capacity (m3/kg VS)",
    )
    nex_override_kg_per_year: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Override N excretion rate (kg N/head/yr)",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source for CO2e conversion",
    )

    @field_validator("manure_allocations")
    @classmethod
    def validate_allocations_sum(
        cls, v: List[ManureSystemAllocation]
    ) -> List[ManureSystemAllocation]:
        """Validate that allocation fractions sum to 1.0."""
        total = sum(a.allocation_fraction for a in v)
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Manure allocation fractions must sum to 1.0 "
                f"(within 1% tolerance), got {total}"
            )
        return v

class CroplandInput(GreenLangBase):
    """Input parameters for agricultural soils N2O calculations.

    Captures nitrogen inputs to managed agricultural soils for
    direct and indirect N2O emission calculations per IPCC 2006
    Volume 4 Chapter 11.

    Attributes:
        id: Unique record identifier (UUID).
        farm_id: Reference to the farm.
        crop_type: Primary crop type grown.
        area_ha: Cropped area in hectares.
        crop_yield_tonnes_per_ha: Crop yield (tonnes/ha).
        synthetic_n_kg: Total synthetic N fertilizer applied (kg N).
        organic_n_kg: Total organic N (manure, compost) applied (kg N).
        crop_residue_n_kg: N returned in crop residues (kg N).
            If not provided, estimated from yield and RPR.
        sewage_sludge_n_kg: N from sewage sludge applied (kg N).
        limestone_tonnes: Limestone applied (tonnes CaCO3).
        dolomite_tonnes: Dolomite applied (tonnes CaMg(CO3)2).
        urea_tonnes: Urea applied (tonnes CO(NH2)2).
        fraction_burned: Fraction of crop residues burned in field.
        soil_type: Soil type for N2O EF adjustment.
        organic_soil_area_ha: Area of organic soils in cropland.
        is_irrigated: Whether cropland is irrigated.
        include_indirect_n2o: Whether to calculate indirect N2O.
        data_quality_tier: Quality tier of input data.
        reference_year: Reference year for the calculation.
        notes: Optional notes.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID)",
    )
    farm_id: str = Field(
        default="",
        description="Reference to the farm",
    )
    crop_type: CropType = Field(
        ...,
        description="Primary crop type grown",
    )
    area_ha: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Cropped area in hectares",
    )
    crop_yield_tonnes_per_ha: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Crop yield (tonnes/ha)",
    )
    synthetic_n_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total synthetic N fertilizer applied (kg N)",
    )
    organic_n_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total organic N applied (kg N)",
    )
    crop_residue_n_kg: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="N returned in crop residues (kg N)",
    )
    sewage_sludge_n_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="N from sewage sludge applied (kg N)",
    )
    limestone_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Limestone applied (tonnes CaCO3)",
    )
    dolomite_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Dolomite applied (tonnes CaMg(CO3)2)",
    )
    urea_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Urea applied (tonnes CO(NH2)2)",
    )
    fraction_burned: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of crop residues burned in field",
    )
    soil_type: SoilType = Field(
        default=SoilType.MINERAL,
        description="Soil type for N2O EF adjustment",
    )
    organic_soil_area_ha: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Area of organic soils in cropland (ha)",
    )
    is_irrigated: bool = Field(
        default=False,
        description="Whether cropland is irrigated",
    )
    include_indirect_n2o: bool = Field(
        default=True,
        description="Whether to calculate indirect N2O emissions",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Quality tier of input data",
    )
    reference_year: int = Field(
        default=2025,
        ge=1900,
        le=2100,
        description="Reference year for the calculation",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes",
    )

class RiceFieldInput(GreenLangBase):
    """Input parameters for rice cultivation CH4 calculations.

    Captures the field-level parameters needed for rice paddy
    methane emission calculations per IPCC 2006 Volume 4 Chapter 5.
    Formula: CH4_rice = EFc * SFw * SFp * SFo * A * t * 1e-6
    where EFc = baseline EF, SFw/SFp/SFo = scaling factors,
    A = area, t = growing season length.

    Attributes:
        id: Unique record identifier (UUID).
        farm_id: Reference to the farm.
        field_name: Human-readable field name.
        area_ha: Rice paddy area in hectares.
        water_regime: Water management regime.
        pre_season_flooding: Pre-season flooding status.
        growing_season_days: Length of growing season (days).
        organic_amendments: List of organic amendment applications.
        organic_amendment_rates: Application rates (tonnes DM/ha)
            matching organic_amendments list.
        baseline_ef_override: Override baseline EF (kg CH4/ha/day).
        cultivations_per_year: Number of rice crops per year.
        data_quality_tier: Quality tier of input data.
        reference_year: Reference year for the calculation.
        notes: Optional notes.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID)",
    )
    farm_id: str = Field(
        default="",
        description="Reference to the farm",
    )
    field_name: str = Field(
        default="",
        max_length=500,
        description="Human-readable field name",
    )
    area_ha: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Rice paddy area in hectares",
    )
    water_regime: WaterRegime = Field(
        default=WaterRegime.CONTINUOUSLY_FLOODED,
        description="Water management regime",
    )
    pre_season_flooding: PreSeasonFlooding = Field(
        default=PreSeasonFlooding.NOT_KNOWN,
        description="Pre-season flooding status",
    )
    growing_season_days: int = Field(
        default=120,
        gt=0,
        le=365,
        description="Length of growing season in days",
    )
    organic_amendments: List[OrganicAmendment] = Field(
        default_factory=list,
        description="List of organic amendment types applied",
    )
    organic_amendment_rates: List[Decimal] = Field(
        default_factory=list,
        description="Application rates (tonnes DM/ha) per amendment",
    )
    baseline_ef_override: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Override baseline EF (kg CH4/ha/day)",
    )
    cultivations_per_year: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Number of rice crops per year",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Quality tier of input data",
    )
    reference_year: int = Field(
        default=2025,
        ge=1900,
        le=2100,
        description="Reference year for the calculation",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes",
    )

    @field_validator("organic_amendment_rates")
    @classmethod
    def validate_amendment_rates(
        cls, v: List[Decimal], info: Any
    ) -> List[Decimal]:
        """Validate amendment rates match amendments list length."""
        data = info.data if hasattr(info, "data") else {}
        amendments = data.get("organic_amendments", [])
        if amendments and len(v) != len(amendments):
            raise ValueError(
                f"organic_amendment_rates length ({len(v)}) must match "
                f"organic_amendments length ({len(amendments)})"
            )
        for rate in v:
            if rate < Decimal("0"):
                raise ValueError(
                    f"Amendment rate must be non-negative, got {rate}"
                )
        return v

class FieldBurningInput(GreenLangBase):
    """Input parameters for field burning of crop residue calculations.

    Captures parameters for calculating CH4 and N2O emissions from
    burning of crop residues in agricultural fields per IPCC 2006
    Volume 4 Chapter 2.

    Attributes:
        id: Unique record identifier (UUID).
        farm_id: Reference to the farm.
        crop_type: Crop type whose residues are burned.
        crop_production_tonnes: Total crop production (tonnes).
        fraction_burned: Fraction of residues burned (0.0-1.0).
        rpr_override: Override residue-to-product ratio.
        dm_override: Override dry matter fraction of residue.
        cf_override: Override combustion factor.
        data_quality_tier: Quality tier of input data.
        reference_year: Reference year for the calculation.
        notes: Optional notes.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID)",
    )
    farm_id: str = Field(
        default="",
        description="Reference to the farm",
    )
    crop_type: CropType = Field(
        ...,
        description="Crop type whose residues are burned",
    )
    crop_production_tonnes: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Total crop production in tonnes",
    )
    fraction_burned: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Fraction of residues burned in field",
    )
    rpr_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Override residue-to-product ratio",
    )
    dm_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        le=Decimal("1.0"),
        description="Override dry matter fraction of residue",
    )
    cf_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        le=Decimal("1.0"),
        description="Override combustion factor",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Quality tier of input data",
    )
    reference_year: int = Field(
        default=2025,
        ge=1900,
        le=2100,
        description="Reference year for the calculation",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Optional notes",
    )

class CalculationRequest(GreenLangBase):
    """Unified request for agricultural emission calculation.

    Specifies all parameters needed to compute GHG emissions from
    one or more agricultural emission sources at a farm. Supports
    enteric fermentation, manure management, agricultural soils,
    rice cultivation, liming/urea, and field burning.

    Attributes:
        id: Unique request identifier (UUID).
        farm_id: Reference to the farm.
        emission_sources: List of emission source categories to
            calculate.
        enteric_requests: Enteric fermentation calculation inputs.
        manure_requests: Manure management calculation inputs.
        cropland_inputs: Agricultural soils N2O inputs.
        rice_field_inputs: Rice cultivation CH4 inputs.
        field_burning_inputs: Field burning inputs.
        calculation_method: Overall calculation methodology.
        gwp_source: GWP source for CO2e conversion.
        data_quality_tier: Overall data quality tier.
        climate_zone: Climate zone for EF stratification.
        include_indirect_n2o: Whether to calculate indirect N2O.
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
    farm_id: str = Field(
        default="",
        description="Reference to the farm",
    )
    emission_sources: List[EmissionSource] = Field(
        default_factory=list,
        description="Emission source categories to calculate",
    )
    enteric_requests: List[EntericCalculationRequest] = Field(
        default_factory=list,
        description="Enteric fermentation calculation inputs",
    )
    manure_requests: List[ManureCalculationRequest] = Field(
        default_factory=list,
        description="Manure management calculation inputs",
    )
    cropland_inputs: List[CroplandInput] = Field(
        default_factory=list,
        description="Agricultural soils N2O inputs",
    )
    rice_field_inputs: List[RiceFieldInput] = Field(
        default_factory=list,
        description="Rice cultivation CH4 inputs",
    )
    field_burning_inputs: List[FieldBurningInput] = Field(
        default_factory=list,
        description="Field burning inputs",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.IPCC_TIER_1,
        description="Overall calculation methodology",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source for CO2e conversion",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Overall data quality tier",
    )
    climate_zone: ClimateZone = Field(
        default=ClimateZone.COOL_TEMPERATE_WET,
        description="Climate zone for EF stratification",
    )
    include_indirect_n2o: bool = Field(
        default=True,
        description="Whether to calculate indirect N2O emissions",
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

class GasEmissionDetail(GreenLangBase):
    """Detailed emission result for a single greenhouse gas.

    Represents the emission of a single gas from an agricultural
    emission calculation, including both mass and CO2-equivalent
    values with the emission factor and formula used.

    Attributes:
        gas: Greenhouse gas species.
        emission_source: Agricultural emission source category.
        emission_mass_tonnes: Emission in tonnes of the gas.
        emission_tco2e: Emission in tonnes CO2 equivalent.
        gwp_applied: GWP value applied for CO2e conversion.
        is_biogenic: Whether emission is biogenic origin.
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
    emission_source: EmissionSource = Field(
        ...,
        description="Agricultural emission source category",
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
        default=True,
        description="Whether emission is biogenic origin",
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

class CalculationResult(GreenLangBase):
    """Complete result of an agricultural emission calculation.

    Attributes:
        id: Unique result identifier (UUID).
        request_id: Reference to the original calculation request.
        total_co2e: Total emissions in tonnes CO2 equivalent.
        emissions_by_gas: Emissions breakdown by gas (tCO2e).
        emissions_by_source: Emissions breakdown by source (tCO2e).
        gas_details: List of detailed per-gas emission results.
        enteric_ch4_tco2e: Enteric fermentation CH4 in tCO2e.
        manure_ch4_tco2e: Manure management CH4 in tCO2e.
        manure_n2o_tco2e: Manure management N2O in tCO2e.
        soil_direct_n2o_tco2e: Soil direct N2O in tCO2e.
        soil_indirect_n2o_tco2e: Soil indirect N2O in tCO2e.
        rice_ch4_tco2e: Rice cultivation CH4 in tCO2e.
        liming_co2_tco2e: Liming CO2 in tCO2e.
        urea_co2_tco2e: Urea application CO2 in tCO2e.
        burning_ch4_tco2e: Field burning CH4 in tCO2e.
        burning_n2o_tco2e: Field burning N2O in tCO2e.
        calculation_method: Calculation method used.
        data_quality_tier: Data quality tier used.
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
    emissions_by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by gas (tCO2e)",
    )
    emissions_by_source: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by source category (tCO2e)",
    )
    gas_details: List[GasEmissionDetail] = Field(
        default_factory=list,
        description="Detailed per-gas emission results",
    )
    enteric_ch4_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Enteric fermentation CH4 in tCO2e",
    )
    manure_ch4_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Manure management CH4 in tCO2e",
    )
    manure_n2o_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Manure management N2O in tCO2e",
    )
    soil_direct_n2o_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Soil direct N2O in tCO2e",
    )
    soil_indirect_n2o_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Soil indirect N2O in tCO2e",
    )
    rice_ch4_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Rice cultivation CH4 in tCO2e",
    )
    liming_co2_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Liming CO2 in tCO2e",
    )
    urea_co2_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Urea application CO2 in tCO2e",
    )
    burning_ch4_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Field burning CH4 in tCO2e",
    )
    burning_n2o_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Field burning N2O in tCO2e",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Calculation method used",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier used",
    )
    trace_steps: List[str] = Field(
        default_factory=list,
        description="Ordered list of calculation trace steps",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
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

class BatchCalculationRequest(GreenLangBase):
    """Batch request for multiple agricultural emission calculations.

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

class BatchCalculationResult(GreenLangBase):
    """Result of a batch agricultural emission calculation.

    Attributes:
        id: Unique batch result identifier (UUID).
        results: List of individual calculation results.
        total_co2e: Aggregate emissions in tonnes CO2e.
        emissions_by_source: Aggregate by emission source (tCO2e).
        emissions_by_gas: Aggregate by gas (tCO2e).
        calculation_count: Number of calculations in the batch.
        failed_count: Number of calculations that failed.
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
    emissions_by_source: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aggregate emissions by source category (tCO2e)",
    )
    emissions_by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aggregate emissions by gas (tCO2e)",
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
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of batch completion",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )

class ComplianceCheckResult(GreenLangBase):
    """Result of a regulatory compliance check.

    Evaluates a calculation or farm against one of the seven
    supported regulatory frameworks (IPCC, GHG Protocol, CSRD,
    EPA, FAO GLEAM, SBTi FLAG, ISO 14064).

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
        farm_id: Reference to the farm checked.
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
    farm_id: str = Field(
        default="",
        description="Reference to the farm checked",
    )
    checked_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the compliance check",
    )

class UncertaintyRequest(GreenLangBase):
    """Request for uncertainty quantification of a calculation.

    Uses Monte Carlo simulation to propagate uncertainty through
    emission factor ranges, activity data variability, and
    parameter uncertainty for agricultural emission calculations.

    Attributes:
        id: Unique request identifier (UUID).
        calculation_id: Reference to the calculation result.
        iterations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.
        confidence_level: Confidence level percentage (e.g. 95.0).
        ef_uncertainty_pct: Emission factor uncertainty as
            percentage (default varies by tier).
        activity_data_uncertainty_pct: Activity data uncertainty
            as percentage.
        population_uncertainty_pct: Livestock population count
            uncertainty as percentage.
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
    activity_data_uncertainty_pct: Decimal = Field(
        default=Decimal("20.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Activity data uncertainty percentage",
    )
    population_uncertainty_pct: Decimal = Field(
        default=Decimal("10.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Livestock population count uncertainty percentage",
    )

class UncertaintyResult(GreenLangBase):
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
        uncertainty_by_source: Uncertainty breakdown by source.
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
    uncertainty_by_source: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Uncertainty contribution by emission source (%)",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of analysis completion",
    )

class AggregationResult(GreenLangBase):
    """Result of an agricultural emission aggregation.

    Attributes:
        id: Unique result identifier (UUID).
        tenant_id: Tenant identifier for scoping.
        groups: Dictionary mapping group keys to aggregated values.
        total_co2e: Total emissions in tonnes CO2e.
        emissions_by_source: Aggregate by emission source (tCO2e).
        emissions_by_gas: Aggregate by gas (tCO2e).
        total_livestock_head: Total livestock head counted.
        total_cropland_ha: Total cropland area in hectares.
        total_rice_ha: Total rice paddy area in hectares.
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
    tenant_id: str = Field(
        default="",
        description="Tenant identifier for scoping",
    )
    groups: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description="Group keys mapped to aggregated values",
    )
    total_co2e: Decimal = Field(
        default=Decimal("0"),
        description="Total emissions in tonnes CO2e",
    )
    emissions_by_source: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aggregate emissions by source category (tCO2e)",
    )
    emissions_by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aggregate emissions by gas (tCO2e)",
    )
    total_livestock_head: int = Field(
        default=0,
        ge=0,
        description="Total livestock head counted",
    )
    total_cropland_ha: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total cropland area in hectares",
    )
    total_rice_ha: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total rice paddy area in hectares",
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
        default_factory=utcnow,
        description="UTC timestamp of aggregation completion",
    )

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "AnimalType",
    "ManureSystem",
    "CropType",
    "FertilizerType",
    "WaterRegime",
    "OrganicAmendment",
    "CalculationMethod",
    "EmissionGas",
    "GWPSource",
    "EmissionFactorSource",
    "DataQualityTier",
    "FarmType",
    "ClimateZone",
    "EmissionSource",
    "ComplianceStatus",
    "ReportingPeriod",
    "PreSeasonFlooding",
    "SoilType",
    # Constant tables
    "GWP_VALUES",
    "CONVERSION_C_TO_CO2",
    "CONVERSION_N_TO_N2O",
    "CH4_C_RATIO",
    "CH4_DENSITY_STP",
    "ENTERIC_EF_TIER1",
    "MANURE_VS_DEFAULTS",
    "MANURE_BO_VALUES",
    "MANURE_NEX_VALUES",
    "MANURE_MCF_VALUES",
    "MANURE_N2O_EF",
    "SOIL_N2O_EF",
    "INDIRECT_N2O_FRACTIONS",
    "LIMING_EF",
    "UREA_EF",
    "RICE_BASELINE_EF",
    "RICE_WATER_REGIME_SF",
    "RICE_PRESEASON_SF",
    "RICE_ORGANIC_CFOA",
    "FIELD_BURNING_EF",
    # Data models
    "FarmInfo",
    "LivestockPopulation",
    "ManureSystemAllocation",
    "FeedCharacteristics",
    "EntericCalculationRequest",
    "ManureCalculationRequest",
    "CroplandInput",
    "RiceFieldInput",
    "FieldBurningInput",
    "CalculationRequest",
    "GasEmissionDetail",
    "CalculationResult",
    "BatchCalculationRequest",
    "BatchCalculationResult",
    "ComplianceCheckResult",
    "UncertaintyRequest",
    "UncertaintyResult",
    "AggregationResult",
    # Scalar constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_LIVESTOCK_PER_CALC",
    "MAX_FARMS_PER_TENANT",
    "DEFAULT_VS_FRACTION",
    "DEFAULT_MCF",
    "DEFAULT_OXIDATION_FACTOR",
    # Helper
    "_utcnow",
]
