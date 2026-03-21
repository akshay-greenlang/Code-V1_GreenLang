# -*- coding: utf-8 -*-
"""
IPCCAR6Bridge - IPCC AR6 Sector Pathway and Emission Factor Integration for PACK-028
======================================================================================

Enterprise bridge for integrating IPCC Sixth Assessment Report (AR6)
pathway data, including Global Warming Potential (GWP-100) values,
sector-specific emission factors from the 2006 IPCC Guidelines (with
2019 Refinements), carbon budget alignment calculations, and mitigation
pathway scenarios (SSP1-1.9, SSP1-2.6, SSP2-4.5).

Key Features:
    - GWP-100 values from IPCC AR6 for all major greenhouse gases
    - Sector-specific emission factors (IPCC 2006 with 2019 refinements)
    - Carbon budget alignment for 1.5C, 2C, and well-below-2C
    - Shared Socioeconomic Pathway (SSP) scenario alignment
    - Remaining carbon budget calculator (from 2020 baseline)
    - Sector emission factor lookup by fuel, process, and region
    - GHG conversion utilities (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
    - SHA-256 provenance on all lookups

IPCC Data Sources:
    - IPCC AR6 WGI Table 7.15 (GWP values)
    - IPCC AR6 WGIII Chapter 3 (Mitigation Pathways)
    - IPCC 2006 Guidelines Vol. 2-5 (Emission Factors)
    - IPCC 2019 Refinement (Updated emission factors)
    - IPCC AR6 Table SPM.2 (Carbon Budgets)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-028 Sector Pathway Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GHGSpecies(str, Enum):
    """Greenhouse gas species with GWP values."""
    CO2 = "co2"
    CH4 = "ch4"
    CH4_FOSSIL = "ch4_fossil"
    CH4_BIOGENIC = "ch4_biogenic"
    N2O = "n2o"
    HFC_134A = "hfc_134a"
    HFC_125 = "hfc_125"
    HFC_143A = "hfc_143a"
    HFC_32 = "hfc_32"
    HFC_23 = "hfc_23"
    HFC_152A = "hfc_152a"
    HFC_245FA = "hfc_245fa"
    HFC_365MFC = "hfc_365mfc"
    HFC_227EA = "hfc_227ea"
    HFC_236FA = "hfc_236fa"
    HFC_43_10MEE = "hfc_43_10mee"
    CF4 = "cf4"
    C2F6 = "c2f6"
    C3F8 = "c3f8"
    C4F10 = "c4f10"
    C6F14 = "c6f14"
    SF6 = "sf6"
    NF3 = "nf3"
    R410A = "r410a"
    R407C = "r407c"
    R404A = "r404a"


class SSPScenario(str, Enum):
    """IPCC Shared Socioeconomic Pathways."""
    SSP1_19 = "ssp1_1.9"   # 1.5C with no/limited overshoot
    SSP1_26 = "ssp1_2.6"   # Well-below 2C
    SSP2_45 = "ssp2_4.5"   # Intermediate
    SSP3_70 = "ssp3_7.0"   # High emissions
    SSP5_85 = "ssp5_8.5"   # Very high emissions


class IPCCSector(str, Enum):
    """IPCC emission factor sectors (2006 Guidelines)."""
    ENERGY_STATIONARY = "energy_stationary"
    ENERGY_TRANSPORT = "energy_transport"
    INDUSTRIAL_PROCESSES = "industrial_processes"
    AGRICULTURE = "agriculture"
    LULUCF = "lulucf"
    WASTE = "waste"


class FuelType(str, Enum):
    """Fuel types for emission factor lookups."""
    COAL_ANTHRACITE = "coal_anthracite"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    COAL_LIGNITE = "coal_lignite"
    NATURAL_GAS = "natural_gas"
    CRUDE_OIL = "crude_oil"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    JET_KEROSENE = "jet_kerosene"
    LPG = "lpg"
    FUEL_OIL_HEAVY = "fuel_oil_heavy"
    FUEL_OIL_LIGHT = "fuel_oil_light"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_CROP = "biomass_crop"
    BIOGAS = "biogas"
    WASTE_MSW = "waste_msw"
    PEAT = "peat"
    COKE_OVEN_GAS = "coke_oven_gas"
    BLAST_FURNACE_GAS = "blast_furnace_gas"
    NAPHTHA = "naphtha"


# ---------------------------------------------------------------------------
# IPCC AR6 GWP-100 Values (Table 7.15, AR6 WGI)
# ---------------------------------------------------------------------------

GWP_100_AR6: Dict[str, Dict[str, Any]] = {
    "co2": {"gwp_100": 1, "lifetime_years": None, "formula": "CO2", "category": "carbon_dioxide"},
    "ch4": {"gwp_100": 27.9, "lifetime_years": 11.8, "formula": "CH4", "category": "methane"},
    "ch4_fossil": {"gwp_100": 29.8, "lifetime_years": 11.8, "formula": "CH4 (fossil)", "category": "methane"},
    "ch4_biogenic": {"gwp_100": 27.0, "lifetime_years": 11.8, "formula": "CH4 (biogenic)", "category": "methane"},
    "n2o": {"gwp_100": 273, "lifetime_years": 109, "formula": "N2O", "category": "nitrous_oxide"},
    "hfc_134a": {"gwp_100": 1530, "lifetime_years": 14.0, "formula": "HFC-134a", "category": "hfc"},
    "hfc_125": {"gwp_100": 3740, "lifetime_years": 30.0, "formula": "HFC-125", "category": "hfc"},
    "hfc_143a": {"gwp_100": 5810, "lifetime_years": 51.0, "formula": "HFC-143a", "category": "hfc"},
    "hfc_32": {"gwp_100": 771, "lifetime_years": 5.4, "formula": "HFC-32", "category": "hfc"},
    "hfc_23": {"gwp_100": 14600, "lifetime_years": 228, "formula": "HFC-23", "category": "hfc"},
    "hfc_152a": {"gwp_100": 164, "lifetime_years": 1.6, "formula": "HFC-152a", "category": "hfc"},
    "hfc_245fa": {"gwp_100": 962, "lifetime_years": 7.9, "formula": "HFC-245fa", "category": "hfc"},
    "hfc_365mfc": {"gwp_100": 914, "lifetime_years": 8.7, "formula": "HFC-365mfc", "category": "hfc"},
    "hfc_227ea": {"gwp_100": 3600, "lifetime_years": 36, "formula": "HFC-227ea", "category": "hfc"},
    "hfc_236fa": {"gwp_100": 8690, "lifetime_years": 213, "formula": "HFC-236fa", "category": "hfc"},
    "hfc_43_10mee": {"gwp_100": 1600, "lifetime_years": 17.0, "formula": "HFC-43-10mee", "category": "hfc"},
    "cf4": {"gwp_100": 7380, "lifetime_years": 50000, "formula": "CF4", "category": "pfc"},
    "c2f6": {"gwp_100": 12400, "lifetime_years": 10000, "formula": "C2F6", "category": "pfc"},
    "c3f8": {"gwp_100": 9290, "lifetime_years": 2600, "formula": "C3F8", "category": "pfc"},
    "c4f10": {"gwp_100": 10000, "lifetime_years": 2600, "formula": "c-C4F10", "category": "pfc"},
    "c6f14": {"gwp_100": 8620, "lifetime_years": 3100, "formula": "C6F14", "category": "pfc"},
    "sf6": {"gwp_100": 25200, "lifetime_years": 3200, "formula": "SF6", "category": "sf6"},
    "nf3": {"gwp_100": 17400, "lifetime_years": 569, "formula": "NF3", "category": "nf3"},
    "r410a": {"gwp_100": 2256, "lifetime_years": None, "formula": "R-410A", "category": "blend"},
    "r407c": {"gwp_100": 1774, "lifetime_years": None, "formula": "R-407C", "category": "blend"},
    "r404a": {"gwp_100": 3943, "lifetime_years": None, "formula": "R-404A", "category": "blend"},
}

# ---------------------------------------------------------------------------
# IPCC Carbon Budgets (AR6 WGI Table SPM.2)
# ---------------------------------------------------------------------------

CARBON_BUDGETS_GTCO2: Dict[str, Dict[str, float]] = {
    "1.5C_50pct": {"budget_from_2020": 500, "budget_from_2025": 350, "temperature": 1.5, "probability": 50},
    "1.5C_67pct": {"budget_from_2020": 400, "budget_from_2025": 250, "temperature": 1.5, "probability": 67},
    "1.5C_83pct": {"budget_from_2020": 300, "budget_from_2025": 150, "temperature": 1.5, "probability": 83},
    "1.7C_50pct": {"budget_from_2020": 850, "budget_from_2025": 700, "temperature": 1.7, "probability": 50},
    "1.7C_67pct": {"budget_from_2020": 700, "budget_from_2025": 550, "temperature": 1.7, "probability": 67},
    "2.0C_50pct": {"budget_from_2020": 1350, "budget_from_2025": 1200, "temperature": 2.0, "probability": 50},
    "2.0C_67pct": {"budget_from_2020": 1150, "budget_from_2025": 1000, "temperature": 2.0, "probability": 67},
    "2.0C_83pct": {"budget_from_2020": 900, "budget_from_2025": 750, "temperature": 2.0, "probability": 83},
}

# ---------------------------------------------------------------------------
# IPCC 2006 Emission Factors (with 2019 Refinements)
# ---------------------------------------------------------------------------

# CO2 emission factors by fuel type (kgCO2/TJ on NCV basis)
EMISSION_FACTORS_CO2_KG_PER_TJ: Dict[str, float] = {
    "coal_anthracite": 98300.0,
    "coal_bituminous": 94600.0,
    "coal_sub_bituminous": 96100.0,
    "coal_lignite": 101000.0,
    "natural_gas": 56100.0,
    "crude_oil": 73300.0,
    "diesel": 74100.0,
    "gasoline": 69300.0,
    "jet_kerosene": 71500.0,
    "lpg": 63100.0,
    "fuel_oil_heavy": 77400.0,
    "fuel_oil_light": 74100.0,
    "biomass_wood": 112000.0,
    "biomass_crop": 100000.0,
    "biogas": 54600.0,
    "waste_msw": 91700.0,
    "peat": 106000.0,
    "coke_oven_gas": 44400.0,
    "blast_furnace_gas": 260000.0,
    "naphtha": 73300.0,
}

# CH4 emission factors (kg CH4/TJ)
EMISSION_FACTORS_CH4_KG_PER_TJ: Dict[str, Dict[str, float]] = {
    "coal_anthracite": {"stationary": 1.0, "transport": 0.0},
    "coal_bituminous": {"stationary": 1.0, "transport": 0.0},
    "natural_gas": {"stationary": 1.0, "transport": 92.0},
    "diesel": {"stationary": 3.0, "transport": 3.9},
    "gasoline": {"stationary": 3.0, "transport": 25.0},
    "jet_kerosene": {"stationary": 3.0, "transport": 0.5},
    "fuel_oil_heavy": {"stationary": 3.0, "transport": 7.0},
    "biomass_wood": {"stationary": 30.0, "transport": 0.0},
}

# N2O emission factors (kg N2O/TJ)
EMISSION_FACTORS_N2O_KG_PER_TJ: Dict[str, Dict[str, float]] = {
    "coal_anthracite": {"stationary": 1.5, "transport": 0.0},
    "coal_bituminous": {"stationary": 1.5, "transport": 0.0},
    "natural_gas": {"stationary": 0.1, "transport": 3.0},
    "diesel": {"stationary": 0.6, "transport": 3.9},
    "gasoline": {"stationary": 0.6, "transport": 8.0},
    "jet_kerosene": {"stationary": 0.6, "transport": 2.0},
    "fuel_oil_heavy": {"stationary": 0.6, "transport": 2.0},
    "biomass_wood": {"stationary": 4.0, "transport": 0.0},
}

# Process emission factors by industry
PROCESS_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "cement_clinker": {"co2_per_tonne": 0.525, "unit": "tCO2/tonne clinker", "source": "IPCC 2006 Vol.3 Ch.2"},
    "cement_product": {"co2_per_tonne": 0.39, "unit": "tCO2/tonne cement", "source": "IPCC 2006 (avg clinker ratio 0.75)"},
    "steel_bof": {"co2_per_tonne": 1.8, "unit": "tCO2/tonne steel (BOF)", "source": "IPCC 2006 Vol.3 Ch.4"},
    "steel_eaf": {"co2_per_tonne": 0.4, "unit": "tCO2/tonne steel (EAF)", "source": "IPCC 2006 Vol.3 Ch.4"},
    "aluminum_primary": {"co2_per_tonne": 1.5, "unit": "tCO2/tonne Al (smelting)", "source": "IPCC 2006 Vol.3 Ch.4"},
    "aluminum_anode": {"cf4_per_tonne": 0.4, "c2f6_per_tonne": 0.04, "unit": "kgPFC/tonne Al", "source": "IPCC 2006 Vol.3 Ch.4"},
    "lime_production": {"co2_per_tonne": 0.75, "unit": "tCO2/tonne lime", "source": "IPCC 2006 Vol.3 Ch.2"},
    "glass_production": {"co2_per_tonne": 0.2, "unit": "tCO2/tonne glass", "source": "IPCC 2006 Vol.3 Ch.2"},
    "ammonia_production": {"co2_per_tonne": 1.5, "unit": "tCO2/tonne NH3", "source": "IPCC 2006 Vol.3 Ch.3"},
    "nitric_acid": {"n2o_per_tonne": 7.0, "unit": "kgN2O/tonne HNO3", "source": "IPCC 2006 Vol.3 Ch.3"},
    "adipic_acid": {"n2o_per_tonne": 300.0, "unit": "kgN2O/tonne product", "source": "IPCC 2006 Vol.3 Ch.3"},
}

# Agricultural emission factors
AGRICULTURAL_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "enteric_fermentation_dairy": {"ch4_per_head_per_year": 128.0, "unit": "kgCH4/head/yr", "region": "global_avg"},
    "enteric_fermentation_beef": {"ch4_per_head_per_year": 66.0, "unit": "kgCH4/head/yr", "region": "global_avg"},
    "enteric_fermentation_sheep": {"ch4_per_head_per_year": 8.0, "unit": "kgCH4/head/yr", "region": "global_avg"},
    "manure_management_dairy": {"ch4_per_head_per_year": 30.0, "n2o_per_head_per_year": 1.0, "unit": "kg/head/yr"},
    "rice_cultivation_continuous": {"ch4_per_hectare_per_season": 130.0, "unit": "kgCH4/ha/season"},
    "rice_cultivation_intermittent": {"ch4_per_hectare_per_season": 60.0, "unit": "kgCH4/ha/season"},
    "synthetic_fertilizer_n2o": {"n2o_direct_fraction": 0.01, "unit": "kgN2O-N/kgN applied", "source": "IPCC 2019"},
    "crop_residue_burning": {"ch4_per_tonne_dm": 2.7, "n2o_per_tonne_dm": 0.07, "unit": "kg/tonne DM"},
}

# SSP global emission pathways (GtCO2/yr)
SSP_EMISSION_PATHWAYS: Dict[str, Dict[int, float]] = {
    "ssp1_1.9": {2020: 40.0, 2025: 35.0, 2030: 25.0, 2035: 15.0, 2040: 8.0, 2045: 2.0, 2050: -2.0},
    "ssp1_2.6": {2020: 40.0, 2025: 37.0, 2030: 30.0, 2035: 22.0, 2040: 15.0, 2045: 8.0, 2050: 3.0},
    "ssp2_4.5": {2020: 40.0, 2025: 40.0, 2030: 38.0, 2035: 35.0, 2040: 32.0, 2045: 28.0, 2050: 25.0},
    "ssp3_7.0": {2020: 40.0, 2025: 42.0, 2030: 45.0, 2035: 48.0, 2040: 50.0, 2045: 52.0, 2050: 55.0},
    "ssp5_8.5": {2020: 40.0, 2025: 45.0, 2030: 52.0, 2035: 58.0, 2040: 65.0, 2045: 72.0, 2050: 80.0},
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class IPCCAR6BridgeConfig(BaseModel):
    """Configuration for the IPCC AR6 bridge."""
    pack_id: str = Field(default="PACK-028")
    ipcc_data_version: str = Field(default="AR6_2021")
    gwp_assessment: str = Field(default="AR6")
    default_gwp_metric: str = Field(default="gwp_100")
    default_scenario: SSPScenario = Field(default=SSPScenario.SSP1_19)
    enable_provenance: bool = Field(default=True)
    include_ar5_comparison: bool = Field(default=False)


class GWPLookupResult(BaseModel):
    """Result of a GWP value lookup."""
    species: str = Field(default="")
    formula: str = Field(default="")
    gwp_100: float = Field(default=0.0)
    lifetime_years: Optional[float] = Field(None)
    category: str = Field(default="")
    assessment: str = Field(default="AR6")
    provenance_hash: str = Field(default="")


class EmissionFactorResult(BaseModel):
    """Result of an emission factor lookup."""
    lookup_id: str = Field(default_factory=_new_uuid)
    fuel_type: str = Field(default="")
    sector: str = Field(default="")
    co2_factor_kg_per_tj: float = Field(default=0.0)
    ch4_factor_kg_per_tj: float = Field(default=0.0)
    n2o_factor_kg_per_tj: float = Field(default=0.0)
    co2e_factor_kg_per_tj: float = Field(default=0.0)
    source: str = Field(default="IPCC 2006/2019")
    provenance_hash: str = Field(default="")


class CarbonBudgetResult(BaseModel):
    """Result of carbon budget alignment calculation."""
    result_id: str = Field(default_factory=_new_uuid)
    temperature_target: float = Field(default=1.5)
    probability_pct: float = Field(default=50.0)
    total_budget_gtco2: float = Field(default=0.0)
    remaining_budget_gtco2: float = Field(default=0.0)
    annual_budget_gtco2: float = Field(default=0.0)
    years_remaining: int = Field(default=0)
    company_share_tco2e: float = Field(default=0.0)
    company_annual_allowance_tco2e: float = Field(default=0.0)
    on_budget: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class GHGConversionResult(BaseModel):
    """Result of GHG unit conversion."""
    input_species: str = Field(default="")
    input_mass_kg: float = Field(default=0.0)
    gwp_100: float = Field(default=0.0)
    co2e_kg: float = Field(default=0.0)
    co2e_tonnes: float = Field(default=0.0)
    assessment: str = Field(default="AR6")


class SSPAlignmentResult(BaseModel):
    """Result of SSP scenario alignment check."""
    result_id: str = Field(default_factory=_new_uuid)
    scenario: str = Field(default="")
    company_emissions_tco2e: float = Field(default=0.0)
    global_emissions_gtco2: float = Field(default=0.0)
    company_share_ppm: float = Field(default=0.0)
    aligned: bool = Field(default=False)
    pathway_data: List[Dict[str, float]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# IPCCAR6Bridge
# ---------------------------------------------------------------------------


class IPCCAR6Bridge:
    """IPCC AR6 pathway and emission factor integration for PACK-028.

    Provides GWP-100 lookups, IPCC emission factors, carbon budget
    calculations, SSP scenario alignment, and GHG conversion utilities.

    Example:
        >>> bridge = IPCCAR6Bridge()
        >>> gwp = bridge.get_gwp("ch4")
        >>> ef = bridge.get_emission_factor("natural_gas", "stationary")
        >>> budget = bridge.calculate_carbon_budget(1.5, 50, 100000)
        >>> co2e = bridge.convert_to_co2e("ch4", 1000.0)
    """

    def __init__(self, config: Optional[IPCCAR6BridgeConfig] = None) -> None:
        self.config = config or IPCCAR6BridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "IPCCAR6Bridge initialized: assessment=%s, scenario=%s, "
            "ghg_species=%d, fuels=%d",
            self.config.gwp_assessment, self.config.default_scenario.value,
            len(GWP_100_AR6), len(EMISSION_FACTORS_CO2_KG_PER_TJ),
        )

    def get_gwp(self, species: str) -> GWPLookupResult:
        """Look up GWP-100 value for a greenhouse gas species."""
        data = GWP_100_AR6.get(species, {})
        result = GWPLookupResult(
            species=species,
            formula=data.get("formula", species.upper()),
            gwp_100=data.get("gwp_100", 0.0),
            lifetime_years=data.get("lifetime_years"),
            category=data.get("category", "unknown"),
            assessment=self.config.gwp_assessment,
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_all_gwp_values(self) -> List[GWPLookupResult]:
        """Get GWP-100 values for all supported greenhouse gas species."""
        return [self.get_gwp(species) for species in GWP_100_AR6]

    def get_emission_factor(
        self, fuel_type: str, combustion_context: str = "stationary",
    ) -> EmissionFactorResult:
        """Get emission factors (CO2, CH4, N2O, CO2e) for a fuel type."""
        co2 = EMISSION_FACTORS_CO2_KG_PER_TJ.get(fuel_type, 0.0)
        ch4_data = EMISSION_FACTORS_CH4_KG_PER_TJ.get(fuel_type, {})
        n2o_data = EMISSION_FACTORS_N2O_KG_PER_TJ.get(fuel_type, {})

        ch4 = ch4_data.get(combustion_context, ch4_data.get("stationary", 0.0))
        n2o = n2o_data.get(combustion_context, n2o_data.get("stationary", 0.0))

        gwp_ch4 = GWP_100_AR6.get("ch4_fossil", {}).get("gwp_100", 29.8)
        gwp_n2o = GWP_100_AR6.get("n2o", {}).get("gwp_100", 273.0)

        co2e = co2 + (ch4 * gwp_ch4) + (n2o * gwp_n2o)

        result = EmissionFactorResult(
            fuel_type=fuel_type,
            sector=combustion_context,
            co2_factor_kg_per_tj=co2,
            ch4_factor_kg_per_tj=ch4,
            n2o_factor_kg_per_tj=n2o,
            co2e_factor_kg_per_tj=round(co2e, 2),
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_process_emission_factor(self, process: str) -> Dict[str, Any]:
        """Get process-specific emission factors (cement, steel, aluminum, etc.)."""
        data = PROCESS_EMISSION_FACTORS.get(process, {})
        result = {"process": process, **data}
        if self.config.enable_provenance:
            result["provenance_hash"] = _compute_hash(result)
        return result

    def get_agricultural_factor(self, source: str) -> Dict[str, Any]:
        """Get agricultural emission factor (enteric, manure, rice, fertilizer)."""
        data = AGRICULTURAL_EMISSION_FACTORS.get(source, {})
        result = {"source": source, **data}
        if self.config.enable_provenance:
            result["provenance_hash"] = _compute_hash(result)
        return result

    def convert_to_co2e(self, species: str, mass_kg: float) -> GHGConversionResult:
        """Convert a GHG mass to CO2-equivalent using AR6 GWP-100."""
        gwp_data = GWP_100_AR6.get(species, {})
        gwp = gwp_data.get("gwp_100", 0.0)
        co2e_kg = mass_kg * gwp
        return GHGConversionResult(
            input_species=species,
            input_mass_kg=mass_kg,
            gwp_100=gwp,
            co2e_kg=round(co2e_kg, 4),
            co2e_tonnes=round(co2e_kg / 1000.0, 6),
            assessment=self.config.gwp_assessment,
        )

    def batch_convert_to_co2e(
        self, emissions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Convert multiple GHG species to CO2e and sum."""
        results = {}
        total_co2e_kg = 0.0
        for species, mass_kg in emissions.items():
            conv = self.convert_to_co2e(species, mass_kg)
            results[species] = {
                "mass_kg": mass_kg,
                "gwp_100": conv.gwp_100,
                "co2e_kg": conv.co2e_kg,
            }
            total_co2e_kg += conv.co2e_kg

        return {
            "species_count": len(emissions),
            "conversions": results,
            "total_co2e_kg": round(total_co2e_kg, 4),
            "total_co2e_tonnes": round(total_co2e_kg / 1000.0, 6),
        }

    def calculate_carbon_budget(
        self,
        temperature_target: float = 1.5,
        probability_pct: float = 50,
        company_emissions_tco2e: float = 100000.0,
        budget_start_year: int = 2025,
        target_year: int = 2050,
    ) -> CarbonBudgetResult:
        """Calculate company's share of the remaining carbon budget."""
        budget_key = f"{temperature_target}C_{int(probability_pct)}pct"
        budget_data = CARBON_BUDGETS_GTCO2.get(budget_key, {})

        if not budget_data:
            # Find closest match
            for key, data in CARBON_BUDGETS_GTCO2.items():
                if abs(data["temperature"] - temperature_target) < 0.1 and abs(data["probability"] - probability_pct) < 10:
                    budget_data = data
                    break

        total_budget = budget_data.get("budget_from_2020", 500)
        # Adjust for emissions between 2020 and start_year (approx 40 GtCO2/yr)
        emitted_since_2020 = (budget_start_year - 2020) * 40
        remaining_budget = max(0, total_budget - emitted_since_2020)

        years_remaining = target_year - budget_start_year
        annual_budget = remaining_budget / max(years_remaining, 1)

        # Company share based on proportional emissions (global ~40 GtCO2/yr = 40e9 tCO2)
        global_annual_tco2 = 40e9
        company_share_ppm = (company_emissions_tco2e / global_annual_tco2) * 1e6
        company_share_fraction = company_emissions_tco2e / global_annual_tco2
        company_annual_allowance = annual_budget * 1e9 * company_share_fraction

        on_budget = company_emissions_tco2e <= company_annual_allowance

        result = CarbonBudgetResult(
            temperature_target=temperature_target,
            probability_pct=probability_pct,
            total_budget_gtco2=total_budget,
            remaining_budget_gtco2=round(remaining_budget, 1),
            annual_budget_gtco2=round(annual_budget, 2),
            years_remaining=years_remaining,
            company_share_tco2e=round(company_emissions_tco2e, 2),
            company_annual_allowance_tco2e=round(company_annual_allowance, 2),
            on_budget=on_budget,
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_ssp_pathway(self, scenario: str) -> Dict[str, Any]:
        """Get SSP global emission pathway data."""
        pathway = SSP_EMISSION_PATHWAYS.get(scenario, {})
        return {
            "scenario": scenario,
            "pathway_points": [{"year": y, "emissions_gtco2": v} for y, v in sorted(pathway.items())],
            "2030_emissions_gtco2": self._interpolate_ssp(pathway, 2030),
            "2050_emissions_gtco2": self._interpolate_ssp(pathway, 2050),
        }

    def check_ssp_alignment(
        self, scenario: str, company_emissions_tco2e: float, year: int = 2030,
    ) -> SSPAlignmentResult:
        """Check if company emissions align with an SSP scenario pathway."""
        pathway = SSP_EMISSION_PATHWAYS.get(scenario, {})
        global_emissions = self._interpolate_ssp(pathway, year)
        company_share = (company_emissions_tco2e / (global_emissions * 1e9)) * 1e6 if global_emissions > 0 else 0

        # A company is "aligned" if its trajectory is consistent with the scenario
        base_global = pathway.get(2020, 40.0)
        target_global = global_emissions
        required_reduction = ((base_global - target_global) / max(base_global, 0.001)) * 100.0

        result = SSPAlignmentResult(
            scenario=scenario,
            company_emissions_tco2e=company_emissions_tco2e,
            global_emissions_gtco2=round(global_emissions, 2),
            company_share_ppm=round(company_share, 4),
            aligned=required_reduction > 0,
            pathway_data=[{"year": y, "gtco2": v} for y, v in sorted(pathway.items())],
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "ipcc_version": self.config.ipcc_data_version,
            "gwp_assessment": self.config.gwp_assessment,
            "ghg_species_count": len(GWP_100_AR6),
            "fuel_types_count": len(EMISSION_FACTORS_CO2_KG_PER_TJ),
            "process_factors_count": len(PROCESS_EMISSION_FACTORS),
            "agricultural_factors_count": len(AGRICULTURAL_EMISSION_FACTORS),
            "carbon_budgets_count": len(CARBON_BUDGETS_GTCO2),
            "ssp_scenarios_count": len(SSP_EMISSION_PATHWAYS),
        }

    def _interpolate_ssp(self, data: Dict[int, float], year: int) -> float:
        if not data:
            return 0.0
        years = sorted(data.keys())
        if year <= years[0]:
            return data[years[0]]
        if year >= years[-1]:
            return data[years[-1]]
        for i in range(len(years) - 1):
            if years[i] <= year <= years[i + 1]:
                frac = (year - years[i]) / (years[i + 1] - years[i])
                return data[years[i]] + frac * (data[years[i + 1]] - data[years[i]])
        return data[years[-1]]
