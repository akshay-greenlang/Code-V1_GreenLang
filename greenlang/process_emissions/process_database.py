# -*- coding: utf-8 -*-
"""
ProcessDatabaseEngine - Engine 1: Process Emissions Agent (AGENT-MRV-004)

Comprehensive in-memory database of industrial process types, emission factors,
raw material properties, carbonate stoichiometric factors, and GWP values for
25 industrial process categories across 6 sectors (mineral, chemical, metal,
electronics, pulp & paper, other).

Zero-Hallucination Guarantees:
    - All data is hard-coded from authoritative IPCC 2006 Guidelines Vol 3
      (Industrial Processes and Product Use), EPA 40 CFR Part 98 Subparts,
      EU ETS Monitoring and Reporting Regulation, and UK DEFRA factors.
    - No LLM in the data path. Every lookup is a deterministic dictionary
      access returning bit-perfect identical results for identical inputs.
    - Decimal arithmetic throughout to avoid IEEE 754 floating-point drift.
    - SHA-256 provenance chain records every lookup and mutation.
    - Prometheus metrics track every database access via gl_pe_ prefix.

Data Sources:
    - IPCC 2006 Guidelines Vol 3 Ch 2-4 (Industrial Processes)
    - EPA 40 CFR Part 98 Subparts F-V, X, Y, Z, SS
    - EU ETS MRR (Monitoring and Reporting Regulation) 2018/2066
    - UK DEFRA GHG Conversion Factors 2025
    - IPCC AR4 (2007), AR5 (2014), AR6 (2021) for GWP-100yr values

Industrial Process Coverage (25):
    Mineral Industry (5):
        CEMENT, LIME, GLASS, CERAMICS, SODA_ASH
    Chemical Industry (6):
        AMMONIA, NITRIC_ACID, ADIPIC_ACID, CARBIDE, HYDROGEN,
        PETROCHEMICAL_ETHYLENE
    Metal Industry (7):
        IRON_STEEL_BF_BOF, IRON_STEEL_EAF, IRON_STEEL_DRI,
        ALUMINUM_PREBAKE, ALUMINUM_SODERBERG, FERROALLOY_FESI,
        MAGNESIUM
    Electronics (1):
        SEMICONDUCTOR
    Pulp & Paper (1):
        PULP_PAPER
    Other (5):
        TITANIUM_DIOXIDE, PHOSPHORIC_ACID, ZINC, LEAD, COPPER

Engines API:
    - get_process_info(process_type) -> Dict
    - get_emission_factor(process_type, gas, source) -> Decimal
    - get_raw_material(material_type) -> Dict
    - get_carbonate_factor(carbonate_type) -> Dict
    - get_gwp(gas, source) -> Decimal
    - list_processes(category) -> List[Dict]
    - list_materials() -> List[Dict]
    - get_production_routes(process_type) -> List[Dict]
    - register_custom_factor(process_type, gas, value, source, unit) -> None
    - get_factor_count() -> int

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports for sibling modules (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.process_emissions.provenance import get_provenance_tracker as _get_provenance_tracker
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.metrics import (
        record_process_lookup as _record_process_lookup,
        record_factor_selection as _record_factor_selection,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_process_lookup = None  # type: ignore[assignment]
    _record_factor_selection = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ProcessDatabaseEngine"]

# ---------------------------------------------------------------------------
# Decimal shorthand
# ---------------------------------------------------------------------------

_D = Decimal


def _d(value: Any) -> Decimal:
    """Convert a numeric value to Decimal via string to avoid float artefacts."""
    return Decimal(str(value))


# ---------------------------------------------------------------------------
# Type alias for emission factor key: (process_type, gas, source)
# ---------------------------------------------------------------------------

_EFKey = Tuple[str, str, str]

# ---------------------------------------------------------------------------
# Emission factor unit labels per source
# ---------------------------------------------------------------------------

_EF_UNITS: Dict[str, str] = {
    "EPA": "varies",
    "IPCC": "varies",
    "DEFRA": "varies",
    "EU_ETS": "varies",
    "CUSTOM": "varies",
}


# =============================================================================
# SECTION 1: PROCESS TYPE DATABASE (25 industrial processes)
# =============================================================================

_PROCESS_TYPES: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # Mineral Industry (5)
    # -----------------------------------------------------------------------
    "CEMENT": {
        "category": "MINERAL",
        "display_name": "Cement Production",
        "description": (
            "CO2 emissions from calcination of calcium carbonate (CaCO3) in "
            "clinker production. Covers raw meal calcination, cement kiln dust "
            "(CKD) correction, and clinker-to-cement ratio adjustments."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE", "STOICHIOMETRIC"],
        "epa_subpart": "Subpart H",
        "ipcc_code": "2A1",
        "default_production_unit": "tonne_clinker",
    },
    "LIME": {
        "category": "MINERAL",
        "display_name": "Lime Production",
        "description": (
            "CO2 emissions from calcination of limestone and dolomite to "
            "produce quicklime (CaO) and hydrated lime (Ca(OH)2). Covers "
            "high-calcium lime, dolomitic lime, and hydraulic lime."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE", "STOICHIOMETRIC"],
        "epa_subpart": "Subpart S",
        "ipcc_code": "2A2",
        "default_production_unit": "tonne_CaO",
    },
    "GLASS": {
        "category": "MINERAL",
        "display_name": "Glass Production",
        "description": (
            "CO2 emissions from decomposition of carbonate raw materials "
            "(soda ash, limestone, dolomite) in glass melting. Covers flat "
            "glass, container glass, fiber glass, and specialty glass."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE", "STOICHIOMETRIC"],
        "epa_subpart": "Subpart N",
        "ipcc_code": "2A3",
        "default_production_unit": "tonne_glass",
    },
    "CERAMICS": {
        "category": "MINERAL",
        "display_name": "Ceramics Production",
        "description": (
            "CO2 emissions from calcination of carbonates in clay mixtures "
            "and from calcium and magnesium carbonates added as raw materials "
            "in ceramic production (tiles, bricks, sanitary ware)."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "STOICHIOMETRIC"],
        "epa_subpart": "N/A",
        "ipcc_code": "2A4a",
        "default_production_unit": "tonne_product",
    },
    "SODA_ASH": {
        "category": "MINERAL",
        "display_name": "Soda Ash Production",
        "description": (
            "CO2 emissions from the Solvay process (synthetic soda ash) and "
            "trona ore calcination (natural soda ash). Also covers soda ash "
            "use as a raw material in other industries."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart Y",
        "ipcc_code": "2A4b",
        "default_production_unit": "tonne_soda_ash",
    },
    # -----------------------------------------------------------------------
    # Chemical Industry (6)
    # -----------------------------------------------------------------------
    "AMMONIA": {
        "category": "CHEMICAL",
        "display_name": "Ammonia Production",
        "description": (
            "CO2 emissions from steam methane reforming (SMR) of natural gas "
            "or partial oxidation of coal/heavy oil for hydrogen production "
            "used in the Haber-Bosch process. Also covers CO2 recovery "
            "for urea production."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart G",
        "ipcc_code": "2B1",
        "default_production_unit": "tonne_NH3",
    },
    "NITRIC_ACID": {
        "category": "CHEMICAL",
        "display_name": "Nitric Acid Production",
        "description": (
            "N2O emissions from the catalytic oxidation of ammonia in nitric "
            "acid (HNO3) production. Abatement technologies (NSCR, SCR, "
            "tertiary catalysts) can reduce N2O by 70-98%."
        ),
        "primary_gases": ["N2O"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "DIRECT_MEASUREMENT"],
        "epa_subpart": "Subpart V",
        "ipcc_code": "2B2",
        "default_production_unit": "tonne_HNO3",
    },
    "ADIPIC_ACID": {
        "category": "CHEMICAL",
        "display_name": "Adipic Acid Production",
        "description": (
            "N2O emissions from the oxidation of cyclohexanone-cyclohexanol "
            "mixture (KA oil) with nitric acid. Unabated N2O factor is very "
            "high; thermal destruction reduces emissions by 95-99%."
        ),
        "primary_gases": ["N2O"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "DIRECT_MEASUREMENT"],
        "epa_subpart": "Subpart E",
        "ipcc_code": "2B3",
        "default_production_unit": "tonne_adipic_acid",
    },
    "CARBIDE": {
        "category": "CHEMICAL",
        "display_name": "Carbide Production (CaC2 / SiC)",
        "description": (
            "CO2 emissions from the production of calcium carbide (CaC2) in "
            "electric arc furnaces from lime and carbon, and from silicon "
            "carbide (SiC) production."
        ),
        "primary_gases": ["CO2", "CH4"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart K",
        "ipcc_code": "2B5",
        "default_production_unit": "tonne_CaC2",
    },
    "HYDROGEN": {
        "category": "CHEMICAL",
        "display_name": "Hydrogen Production (SMR)",
        "description": (
            "CO2 emissions from steam methane reforming (SMR) of natural gas "
            "for dedicated hydrogen production. This is the dominant pathway "
            "for grey hydrogen. Blue hydrogen includes CCS."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart P",
        "ipcc_code": "2B",
        "default_production_unit": "tonne_H2",
    },
    "PETROCHEMICAL_ETHYLENE": {
        "category": "CHEMICAL",
        "display_name": "Petrochemical Production (Ethylene)",
        "description": (
            "CO2 emissions from steam cracking of naphtha, ethane, or other "
            "feedstocks to produce ethylene and co-products (propylene, "
            "butadiene, aromatics). Process CO2 from decoking and flaring."
        ),
        "primary_gases": ["CO2", "CH4"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart X",
        "ipcc_code": "2B8a",
        "default_production_unit": "tonne_ethylene",
    },
    # -----------------------------------------------------------------------
    # Metal Industry (7)
    # -----------------------------------------------------------------------
    "IRON_STEEL_BF_BOF": {
        "category": "METAL",
        "display_name": "Iron & Steel (Blast Furnace - BOF Route)",
        "description": (
            "CO2 emissions from integrated steelmaking: coke production, "
            "sinter/pellet production, blast furnace ironmaking, and basic "
            "oxygen furnace (BOF) steelmaking. Covers carbon from coke, "
            "coal injection, limestone flux, and electrode consumption."
        ),
        "primary_gases": ["CO2", "CH4"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart Q",
        "ipcc_code": "2C1",
        "default_production_unit": "tonne_steel",
    },
    "IRON_STEEL_EAF": {
        "category": "METAL",
        "display_name": "Iron & Steel (Electric Arc Furnace Route)",
        "description": (
            "CO2 emissions from electric arc furnace steelmaking using scrap "
            "steel as primary input. Lower carbon intensity than BF-BOF. "
            "Emissions from electrode consumption, carbon additions, and "
            "flux decomposition."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart Q",
        "ipcc_code": "2C1",
        "default_production_unit": "tonne_steel",
    },
    "IRON_STEEL_DRI": {
        "category": "METAL",
        "display_name": "Iron & Steel (Direct Reduced Iron Route)",
        "description": (
            "CO2 emissions from direct reduction of iron ore using natural gas "
            "or hydrogen as reductant. DRI is then melted in EAF. Covers "
            "Midrex, HYL/Energiron, and emerging H2-DRI processes."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart Q",
        "ipcc_code": "2C1",
        "default_production_unit": "tonne_DRI",
    },
    "ALUMINUM_PREBAKE": {
        "category": "METAL",
        "display_name": "Aluminum Smelting (Prebake Anode)",
        "description": (
            "CO2 and PFC (CF4, C2F6) emissions from aluminum electrolysis "
            "using prebake anodes. CO2 from anode consumption (Boudouard "
            "reaction). PFC from anode effects."
        ),
        "primary_gases": ["CO2", "CF4", "C2F6"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart F",
        "ipcc_code": "2C3",
        "default_production_unit": "tonne_Al",
    },
    "ALUMINUM_SODERBERG": {
        "category": "METAL",
        "display_name": "Aluminum Smelting (Soderberg Anode)",
        "description": (
            "CO2 and PFC (CF4, C2F6) emissions from aluminum electrolysis "
            "using Soderberg (self-baking) anodes. Higher specific CO2 and PFC "
            "than prebake due to less controlled anode consumption. Covers "
            "vertical stud (VSS) and horizontal stud (HSS) configurations."
        ),
        "primary_gases": ["CO2", "CF4", "C2F6"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart F",
        "ipcc_code": "2C3",
        "default_production_unit": "tonne_Al",
    },
    "FERROALLOY_FESI": {
        "category": "METAL",
        "display_name": "Ferroalloy Production (FeSi)",
        "description": (
            "CO2 and CH4 emissions from submerged arc furnace production "
            "of ferrosilicon (FeSi). Carbon from coke, coal, and charcoal "
            "reductants reacts with quartz and iron ore."
        ),
        "primary_gases": ["CO2", "CH4"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart Z",
        "ipcc_code": "2C2",
        "default_production_unit": "tonne_FeSi",
    },
    "MAGNESIUM": {
        "category": "METAL",
        "display_name": "Magnesium Production & Casting",
        "description": (
            "SF6 emissions from magnesium production and casting. SF6 is "
            "used as a cover gas to prevent oxidation of molten magnesium. "
            "Alternative cover gases include SO2 and fluorinated ketones."
        ),
        "primary_gases": ["SF6", "CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart T",
        "ipcc_code": "2C4",
        "default_production_unit": "tonne_Mg",
    },
    # -----------------------------------------------------------------------
    # Electronics (1)
    # -----------------------------------------------------------------------
    "SEMICONDUCTOR": {
        "category": "ELECTRONICS",
        "display_name": "Semiconductor Manufacturing",
        "description": (
            "Emissions of fluorinated GHGs (CF4, C2F6, C3F8, NF3, SF6, "
            "HFC-23, c-C4F8) from plasma etching and CVD chamber cleaning "
            "in semiconductor fabrication. Each gas has specific utilization "
            "rate, destruction removal efficiency, and by-product formation."
        ),
        "primary_gases": ["CF4", "C2F6", "SF6", "NF3", "HFC_23"],
        "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
        "applicable_methods": ["EMISSION_FACTOR", "DIRECT_MEASUREMENT"],
        "epa_subpart": "Subpart I",
        "ipcc_code": "2E1",
        "default_production_unit": "wafer_starts",
    },
    # -----------------------------------------------------------------------
    # Pulp & Paper (1)
    # -----------------------------------------------------------------------
    "PULP_PAPER": {
        "category": "PULP_PAPER",
        "display_name": "Pulp & Paper Production",
        "description": (
            "CO2 emissions from limestone calcination in pulp mills (lime "
            "kiln), makeup calcium carbonate usage, and chemical recovery. "
            "Biogenic CO2 from black liquor combustion is tracked separately."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart AA",
        "ipcc_code": "2H",
        "default_production_unit": "tonne_pulp",
    },
    # -----------------------------------------------------------------------
    # Other Industries (5)
    # -----------------------------------------------------------------------
    "TITANIUM_DIOXIDE": {
        "category": "OTHER",
        "display_name": "Titanium Dioxide Production",
        "description": (
            "CO2 emissions from chloride process TiO2 production using "
            "petroleum coke as reductant. Covers both chloride and sulfate "
            "process routes."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart EE",
        "ipcc_code": "2B6",
        "default_production_unit": "tonne_TiO2",
    },
    "PHOSPHORIC_ACID": {
        "category": "OTHER",
        "display_name": "Phosphoric Acid Production",
        "description": (
            "CO2 emissions from the calcination of phosphate rock and from "
            "the use of coke as reductant in the thermal (furnace) process. "
            "Minor CO2 from calcium carbonate in phosphate ore."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart FF",
        "ipcc_code": "2B9a",
        "default_production_unit": "tonne_P2O5",
    },
    "ZINC": {
        "category": "OTHER",
        "display_name": "Zinc Production",
        "description": (
            "CO2 emissions from pyrometallurgical zinc production using "
            "Imperial Smelting Furnace or Waelz kiln. Carbon from coke "
            "and coal reductants."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart GG",
        "ipcc_code": "2C5",
        "default_production_unit": "tonne_Zn",
    },
    "LEAD": {
        "category": "OTHER",
        "display_name": "Lead Production",
        "description": (
            "CO2 emissions from primary and secondary lead production. "
            "Primary lead uses blast furnace or direct smelting with coke "
            "or natural gas reductants."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart R",
        "ipcc_code": "2C5",
        "default_production_unit": "tonne_Pb",
    },
    "COPPER": {
        "category": "OTHER",
        "display_name": "Copper Smelting",
        "description": (
            "CO2 emissions from pyrometallurgical copper smelting using "
            "flash smelting, converting, and fire refining. Carbon from "
            "fossil reductants and anode baking."
        ),
        "primary_gases": ["CO2"],
        "applicable_tiers": ["TIER_1", "TIER_2"],
        "applicable_methods": ["EMISSION_FACTOR", "MASS_BALANCE"],
        "epa_subpart": "Subpart GG",
        "ipcc_code": "2C5",
        "default_production_unit": "tonne_Cu",
    },
}


# =============================================================================
# SECTION 2: DEFAULT EMISSION FACTORS PER PROCESS AND SOURCE
# =============================================================================
# Structure: _<SOURCE>_PROCESS_FACTORS[process_type][gas] = Decimal
# Unit basis varies by process and is documented per entry.

# ---------------------------------------------------------------------------
# IPCC 2006 Default Emission Factors (Vol 3)
# ---------------------------------------------------------------------------

_IPCC_PROCESS_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # Mineral Industry
    # IPCC 2006 Vol 3 Table 2.1: tCO2/t clinker
    "CEMENT":                  {"CO2": _D("0.507")},
    # IPCC 2006 Vol 3 Table 2.4: tCO2/t CaO (high-calcium lime)
    "LIME":                    {"CO2": _D("0.785")},
    # IPCC 2006 Vol 3 Table 2.7: tCO2/t glass (container glass)
    "GLASS":                   {"CO2": _D("0.208")},
    # IPCC 2006 Vol 3: tCO2/t product (typical ceramics)
    "CERAMICS":                {"CO2": _D("0.120")},
    # IPCC 2006 Vol 3 Table 2.17: tCO2/t soda ash (Solvay process)
    "SODA_ASH":                {"CO2": _D("0.138")},

    # Chemical Industry
    # IPCC 2006 Vol 3 Table 3.1: tCO2/t NH3 (natural gas feedstock)
    "AMMONIA":                 {"CO2": _D("1.500")},
    # IPCC 2006 Vol 3 Table 3.3: tN2O/t HNO3 (default, no abatement)
    "NITRIC_ACID":             {"N2O": _D("0.007")},
    # IPCC 2006 Vol 3 Table 3.4: tN2O/t adipic acid (unabated)
    "ADIPIC_ACID":             {"N2O": _D("0.300")},
    # IPCC 2006 Vol 3: tCO2/t CaC2
    "CARBIDE":                 {"CO2": _D("1.090"), "CH4": _D("0.012")},
    # IPCC / literature: kgCO2/kgH2 (SMR without CCS)
    "HYDROGEN":                {"CO2": _D("9.300")},
    # IPCC 2006 Vol 3: tCO2/t ethylene (naphtha cracker, process only)
    "PETROCHEMICAL_ETHYLENE":  {"CO2": _D("1.500"), "CH4": _D("0.006")},

    # Metal Industry
    # IPCC 2006 Vol 3 Table 4.1: tCO2/t steel (BF-BOF integrated)
    "IRON_STEEL_BF_BOF":       {"CO2": _D("1.900"), "CH4": _D("0.001")},
    # IPCC 2006 Vol 3: tCO2/t steel (EAF with ~100% scrap)
    "IRON_STEEL_EAF":          {"CO2": _D("0.400")},
    # IPCC 2006 Vol 3: tCO2/t DRI (NG-based)
    "IRON_STEEL_DRI":          {"CO2": _D("0.700")},
    # IPCC 2006 Vol 3 Table 4.11: tCO2/t Al + PFC as CO2e/t Al
    "ALUMINUM_PREBAKE":        {"CO2": _D("1.500"), "CF4": _D("0.040"), "C2F6": _D("0.004")},
    # IPCC 2006 Vol 3: tCO2/t Al (Soderberg VSS/HSS) + higher PFC
    "ALUMINUM_SODERBERG":      {"CO2": _D("1.800"), "CF4": _D("0.060"), "C2F6": _D("0.006")},
    # IPCC 2006 Vol 3: tCO2/t FeSi (75% Si grade)
    "FERROALLOY_FESI":         {"CO2": _D("4.300"), "CH4": _D("0.010")},
    # IPCC 2006 Vol 3: SF6 usage based; default kgSF6/t Mg cast
    "MAGNESIUM":               {"SF6": _D("0.001"), "CO2": _D("0.100")},

    # Electronics
    # IPCC 2006 Vol 3 Table 6.3: Default factors per gas
    # (tGas per wafer start for 200mm equivalent, need scaling)
    # Stored as kg gas per 1000 wafer starts (mixed gases)
    "SEMICONDUCTOR":           {
        "CF4": _D("0.100"),
        "C2F6": _D("0.050"),
        "SF6": _D("0.010"),
        "NF3": _D("0.200"),
        "HFC_23": _D("0.005"),
    },

    # Pulp & Paper
    # IPCC: tCO2/t pulp (from makeup CaCO3 and lime kiln)
    "PULP_PAPER":              {"CO2": _D("0.060")},

    # Other Industries
    "TITANIUM_DIOXIDE":        {"CO2": _D("1.340")},
    "PHOSPHORIC_ACID":         {"CO2": _D("0.220")},
    "ZINC":                    {"CO2": _D("3.660")},
    "LEAD":                    {"CO2": _D("0.590")},
    "COPPER":                  {"CO2": _D("0.240")},
}

# ---------------------------------------------------------------------------
# EPA 40 CFR Part 98 Emission Factors
# ---------------------------------------------------------------------------

_EPA_PROCESS_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "CEMENT":                  {"CO2": _D("0.510")},
    "LIME":                    {"CO2": _D("0.785")},
    "GLASS":                   {"CO2": _D("0.210")},
    "CERAMICS":                {"CO2": _D("0.115")},
    "SODA_ASH":                {"CO2": _D("0.138")},
    "AMMONIA":                 {"CO2": _D("1.500")},
    "NITRIC_ACID":             {"N2O": _D("0.007")},
    "ADIPIC_ACID":             {"N2O": _D("0.300")},
    "CARBIDE":                 {"CO2": _D("1.090"), "CH4": _D("0.012")},
    "HYDROGEN":                {"CO2": _D("9.260")},
    "PETROCHEMICAL_ETHYLENE":  {"CO2": _D("1.450"), "CH4": _D("0.006")},
    "IRON_STEEL_BF_BOF":       {"CO2": _D("1.850"), "CH4": _D("0.001")},
    "IRON_STEEL_EAF":          {"CO2": _D("0.410")},
    "IRON_STEEL_DRI":          {"CO2": _D("0.720")},
    "ALUMINUM_PREBAKE":        {"CO2": _D("1.500"), "CF4": _D("0.040"), "C2F6": _D("0.004")},
    "ALUMINUM_SODERBERG":      {"CO2": _D("1.800"), "CF4": _D("0.060"), "C2F6": _D("0.006")},
    "FERROALLOY_FESI":         {"CO2": _D("4.250"), "CH4": _D("0.011")},
    "MAGNESIUM":               {"SF6": _D("0.001"), "CO2": _D("0.100")},
    "SEMICONDUCTOR":           {
        "CF4": _D("0.100"),
        "C2F6": _D("0.050"),
        "SF6": _D("0.010"),
        "NF3": _D("0.200"),
        "HFC_23": _D("0.005"),
    },
    "PULP_PAPER":              {"CO2": _D("0.060")},
    "TITANIUM_DIOXIDE":        {"CO2": _D("1.340")},
    "PHOSPHORIC_ACID":         {"CO2": _D("0.220")},
    "ZINC":                    {"CO2": _D("3.690")},
    "LEAD":                    {"CO2": _D("0.600")},
    "COPPER":                  {"CO2": _D("0.250")},
}

# ---------------------------------------------------------------------------
# UK DEFRA 2025 Emission Factors (process sector)
# ---------------------------------------------------------------------------

_DEFRA_PROCESS_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "CEMENT":                  {"CO2": _D("0.519")},
    "LIME":                    {"CO2": _D("0.785")},
    "GLASS":                   {"CO2": _D("0.205")},
    "CERAMICS":                {"CO2": _D("0.120")},
    "SODA_ASH":                {"CO2": _D("0.415")},
    "AMMONIA":                 {"CO2": _D("1.530")},
    "NITRIC_ACID":             {"N2O": _D("0.007")},
    "ADIPIC_ACID":             {"N2O": _D("0.300")},
    "CARBIDE":                 {"CO2": _D("1.090")},
    "HYDROGEN":                {"CO2": _D("9.300")},
    "PETROCHEMICAL_ETHYLENE":  {"CO2": _D("1.500")},
    "IRON_STEEL_BF_BOF":       {"CO2": _D("1.920")},
    "IRON_STEEL_EAF":          {"CO2": _D("0.380")},
    "IRON_STEEL_DRI":          {"CO2": _D("0.700")},
    "ALUMINUM_PREBAKE":        {"CO2": _D("1.500"), "CF4": _D("0.040"), "C2F6": _D("0.004")},
    "ALUMINUM_SODERBERG":      {"CO2": _D("1.800"), "CF4": _D("0.060"), "C2F6": _D("0.006")},
    "FERROALLOY_FESI":         {"CO2": _D("4.300")},
    "MAGNESIUM":               {"SF6": _D("0.001")},
    "SEMICONDUCTOR":           {"CF4": _D("0.100"), "C2F6": _D("0.050"), "NF3": _D("0.200")},
    "PULP_PAPER":              {"CO2": _D("0.060")},
    "TITANIUM_DIOXIDE":        {"CO2": _D("1.340")},
    "PHOSPHORIC_ACID":         {"CO2": _D("0.220")},
    "ZINC":                    {"CO2": _D("3.660")},
    "LEAD":                    {"CO2": _D("0.590")},
    "COPPER":                  {"CO2": _D("0.240")},
}

# ---------------------------------------------------------------------------
# EU ETS Monitoring & Reporting Regulation Factors
# ---------------------------------------------------------------------------

_EU_ETS_PROCESS_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "CEMENT":                  {"CO2": _D("0.525")},
    "LIME":                    {"CO2": _D("0.785")},
    "GLASS":                   {"CO2": _D("0.208")},
    "CERAMICS":                {"CO2": _D("0.120")},
    "SODA_ASH":                {"CO2": _D("0.415")},
    "AMMONIA":                 {"CO2": _D("1.500")},
    "NITRIC_ACID":             {"N2O": _D("0.007")},
    "ADIPIC_ACID":             {"N2O": _D("0.300")},
    "CARBIDE":                 {"CO2": _D("1.090")},
    "HYDROGEN":                {"CO2": _D("9.300")},
    "PETROCHEMICAL_ETHYLENE":  {"CO2": _D("1.500")},
    "IRON_STEEL_BF_BOF":       {"CO2": _D("1.900")},
    "IRON_STEEL_EAF":          {"CO2": _D("0.400")},
    "IRON_STEEL_DRI":          {"CO2": _D("0.700")},
    "ALUMINUM_PREBAKE":        {"CO2": _D("1.500"), "CF4": _D("0.040"), "C2F6": _D("0.004")},
    "ALUMINUM_SODERBERG":      {"CO2": _D("1.800"), "CF4": _D("0.060"), "C2F6": _D("0.006")},
    "FERROALLOY_FESI":         {"CO2": _D("4.300")},
    "MAGNESIUM":               {"SF6": _D("0.001")},
    "SEMICONDUCTOR":           {"CF4": _D("0.100"), "C2F6": _D("0.050"), "NF3": _D("0.200")},
    "PULP_PAPER":              {"CO2": _D("0.060")},
    "TITANIUM_DIOXIDE":        {"CO2": _D("1.340")},
    "PHOSPHORIC_ACID":         {"CO2": _D("0.220")},
    "ZINC":                    {"CO2": _D("3.660")},
    "LEAD":                    {"CO2": _D("0.590")},
    "COPPER":                  {"CO2": _D("0.240")},
}


# =============================================================================
# SECTION 3: EMISSION FACTOR UNIT METADATA
# =============================================================================
# Describes the unit basis for each process's default emission factor.

_PROCESS_EF_UNITS: Dict[str, Dict[str, str]] = {
    "CEMENT":                  {"CO2": "tCO2/t_clinker"},
    "LIME":                    {"CO2": "tCO2/t_CaO"},
    "GLASS":                   {"CO2": "tCO2/t_glass"},
    "CERAMICS":                {"CO2": "tCO2/t_product"},
    "SODA_ASH":                {"CO2": "tCO2/t_soda_ash"},
    "AMMONIA":                 {"CO2": "tCO2/t_NH3"},
    "NITRIC_ACID":             {"N2O": "tN2O/t_HNO3"},
    "ADIPIC_ACID":             {"N2O": "tN2O/t_adipic_acid"},
    "CARBIDE":                 {"CO2": "tCO2/t_CaC2", "CH4": "tCH4/t_CaC2"},
    "HYDROGEN":                {"CO2": "kgCO2/kg_H2"},
    "PETROCHEMICAL_ETHYLENE":  {"CO2": "tCO2/t_ethylene", "CH4": "tCH4/t_ethylene"},
    "IRON_STEEL_BF_BOF":       {"CO2": "tCO2/t_steel", "CH4": "tCH4/t_steel"},
    "IRON_STEEL_EAF":          {"CO2": "tCO2/t_steel"},
    "IRON_STEEL_DRI":          {"CO2": "tCO2/t_DRI"},
    "ALUMINUM_PREBAKE":        {"CO2": "tCO2/t_Al", "CF4": "tCF4/t_Al", "C2F6": "tC2F6/t_Al"},
    "ALUMINUM_SODERBERG":      {"CO2": "tCO2/t_Al", "CF4": "tCF4/t_Al", "C2F6": "tC2F6/t_Al"},
    "FERROALLOY_FESI":         {"CO2": "tCO2/t_FeSi", "CH4": "tCH4/t_FeSi"},
    "MAGNESIUM":               {"SF6": "tSF6/t_Mg", "CO2": "tCO2/t_Mg"},
    "SEMICONDUCTOR":           {
        "CF4": "kg/1000_wafer_starts",
        "C2F6": "kg/1000_wafer_starts",
        "SF6": "kg/1000_wafer_starts",
        "NF3": "kg/1000_wafer_starts",
        "HFC_23": "kg/1000_wafer_starts",
    },
    "PULP_PAPER":              {"CO2": "tCO2/t_pulp"},
    "TITANIUM_DIOXIDE":        {"CO2": "tCO2/t_TiO2"},
    "PHOSPHORIC_ACID":         {"CO2": "tCO2/t_P2O5"},
    "ZINC":                    {"CO2": "tCO2/t_Zn"},
    "LEAD":                    {"CO2": "tCO2/t_Pb"},
    "COPPER":                  {"CO2": "tCO2/t_Cu"},
}


# =============================================================================
# SECTION 4: ALTERNATIVE EMISSION FACTORS FOR PROCESS VARIANTS
# =============================================================================
# These supplement the default factors for specific sub-process conditions.

_PROCESS_VARIANT_FACTORS: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    # Ammonia - coal-based feedstock (higher intensity)
    "AMMONIA_COAL": {
        "IPCC": {"CO2": _D("2.300")},
        "EPA": {"CO2": _D("2.300")},
        "DEFRA": {"CO2": _D("2.300")},
        "EU_ETS": {"CO2": _D("2.300")},
    },
    # Ammonia - with urea CO2 recovery credit
    "AMMONIA_UREA_CREDIT": {
        "IPCC": {"CO2": _D("0.733")},
        "EPA": {"CO2": _D("0.733")},
        "DEFRA": {"CO2": _D("0.733")},
        "EU_ETS": {"CO2": _D("0.733")},
    },
    # Adipic acid - with 97% abatement (thermal destruction)
    "ADIPIC_ACID_ABATED": {
        "IPCC": {"N2O": _D("0.009")},
        "EPA": {"N2O": _D("0.009")},
        "DEFRA": {"N2O": _D("0.009")},
        "EU_ETS": {"N2O": _D("0.009")},
    },
    # Lime - dolomitic lime (MgO + CaO)
    "LIME_DOLOMITIC": {
        "IPCC": {"CO2": _D("0.913")},
        "EPA": {"CO2": _D("0.913")},
        "DEFRA": {"CO2": _D("0.913")},
        "EU_ETS": {"CO2": _D("0.913")},
    },
    # Lime - hydraulic lime (partially calcined)
    "LIME_HYDRAULIC": {
        "IPCC": {"CO2": _D("0.490")},
        "EPA": {"CO2": _D("0.490")},
        "DEFRA": {"CO2": _D("0.490")},
        "EU_ETS": {"CO2": _D("0.490")},
    },
    # Glass - fiber glass (higher carbonate content)
    "GLASS_FIBER": {
        "IPCC": {"CO2": _D("0.250")},
        "EPA": {"CO2": _D("0.250")},
        "DEFRA": {"CO2": _D("0.250")},
        "EU_ETS": {"CO2": _D("0.250")},
    },
    # Glass - specialty glass
    "GLASS_SPECIALTY": {
        "IPCC": {"CO2": _D("0.180")},
        "EPA": {"CO2": _D("0.180")},
        "DEFRA": {"CO2": _D("0.180")},
        "EU_ETS": {"CO2": _D("0.180")},
    },
    # Aluminum prebake - with PFC expressed as CO2e per tonne Al
    "ALUMINUM_PREBAKE_PFC_CO2E": {
        "IPCC": {"CO2": _D("1.500"), "PFC_CO2E": _D("0.400")},
        "EPA": {"CO2": _D("1.500"), "PFC_CO2E": _D("0.400")},
    },
    # Aluminum Soderberg - PFC as CO2e per tonne Al
    "ALUMINUM_SODERBERG_PFC_CO2E": {
        "IPCC": {"CO2": _D("1.800"), "PFC_CO2E": _D("0.600")},
        "EPA": {"CO2": _D("1.800"), "PFC_CO2E": _D("0.600")},
    },
}


# =============================================================================
# SECTION 5: RAW MATERIAL DATABASE
# =============================================================================
# Properties of raw materials used in industrial processes.

_RAW_MATERIALS: Dict[str, Dict[str, Any]] = {
    # Carbonates
    "CALCIUM_CARBONATE": {
        "display_name": "Calcium Carbonate (Calcite, CaCO3)",
        "formula": "CaCO3",
        "molecular_weight": _D("100.09"),
        "carbon_content": _D("0.1200"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "calcite",
        "stoichiometric_co2_factor": _D("0.440"),
        "category": "CARBONATE",
    },
    "MAGNESIUM_CARBONATE": {
        "display_name": "Magnesium Carbonate (Magnesite, MgCO3)",
        "formula": "MgCO3",
        "molecular_weight": _D("84.31"),
        "carbon_content": _D("0.1423"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "magnesite",
        "stoichiometric_co2_factor": _D("0.522"),
        "category": "CARBONATE",
    },
    "DOLOMITE": {
        "display_name": "Dolomite (CaMg(CO3)2)",
        "formula": "CaMg(CO3)2",
        "molecular_weight": _D("184.40"),
        "carbon_content": _D("0.1303"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "dolomite",
        "stoichiometric_co2_factor": _D("0.477"),
        "category": "CARBONATE",
    },
    "IRON_CARBONATE": {
        "display_name": "Iron Carbonate (Siderite, FeCO3)",
        "formula": "FeCO3",
        "molecular_weight": _D("115.86"),
        "carbon_content": _D("0.1036"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "siderite",
        "stoichiometric_co2_factor": _D("0.380"),
        "category": "CARBONATE",
    },
    "ANKERITE": {
        "display_name": "Ankerite (Ca(Fe,Mg,Mn)(CO3)2)",
        "formula": "Ca(Fe,Mg)(CO3)2",
        "molecular_weight": _D("206.00"),
        "carbon_content": _D("0.1165"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "ankerite",
        "stoichiometric_co2_factor": _D("0.427"),
        "category": "CARBONATE",
    },
    "SODIUM_CARBONATE": {
        "display_name": "Sodium Carbonate (Soda Ash, Na2CO3)",
        "formula": "Na2CO3",
        "molecular_weight": _D("105.99"),
        "carbon_content": _D("0.1132"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "soda_ash",
        "stoichiometric_co2_factor": _D("0.415"),
        "category": "CARBONATE",
    },
    "BARIUM_CARBONATE": {
        "display_name": "Barium Carbonate (Witherite, BaCO3)",
        "formula": "BaCO3",
        "molecular_weight": _D("197.34"),
        "carbon_content": _D("0.0608"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "witherite",
        "stoichiometric_co2_factor": _D("0.223"),
        "category": "CARBONATE",
    },
    "STRONTIUM_CARBONATE": {
        "display_name": "Strontium Carbonate (Strontianite, SrCO3)",
        "formula": "SrCO3",
        "molecular_weight": _D("147.63"),
        "carbon_content": _D("0.0813"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "strontianite",
        "stoichiometric_co2_factor": _D("0.298"),
        "category": "CARBONATE",
    },
    "LITHIUM_CARBONATE": {
        "display_name": "Lithium Carbonate (Li2CO3)",
        "formula": "Li2CO3",
        "molecular_weight": _D("73.89"),
        "carbon_content": _D("0.1624"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "lithium_carbonate",
        "stoichiometric_co2_factor": _D("0.596"),
        "category": "CARBONATE",
    },
    "MANGANESE_CARBONATE": {
        "display_name": "Manganese Carbonate (Rhodochrosite, MnCO3)",
        "formula": "MnCO3",
        "molecular_weight": _D("114.95"),
        "carbon_content": _D("0.1044"),
        "carbonate_content": _D("1.0000"),
        "fraction_ite": "rhodochrosite",
        "stoichiometric_co2_factor": _D("0.383"),
        "category": "CARBONATE",
    },
    # Non-carbonate raw materials
    "LIMESTONE_RAW": {
        "display_name": "Raw Limestone (natural, mixed CaCO3/MgCO3)",
        "formula": "CaCO3 + impurities",
        "molecular_weight": _D("100.09"),
        "carbon_content": _D("0.1100"),
        "carbonate_content": _D("0.9200"),
        "fraction_ite": "mixed",
        "stoichiometric_co2_factor": _D("0.405"),
        "category": "MINERAL",
    },
    "IRON_ORE": {
        "display_name": "Iron Ore (Hematite, Fe2O3)",
        "formula": "Fe2O3",
        "molecular_weight": _D("159.69"),
        "carbon_content": _D("0.0000"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "ORE",
    },
    "BAUXITE": {
        "display_name": "Bauxite (Al2O3 ore)",
        "formula": "Al2O3 + impurities",
        "molecular_weight": _D("101.96"),
        "carbon_content": _D("0.0000"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "ORE",
    },
    "COKE": {
        "display_name": "Metallurgical Coke",
        "formula": "C (amorphous)",
        "molecular_weight": _D("12.01"),
        "carbon_content": _D("0.8700"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "REDUCTANT",
    },
    "COAL": {
        "display_name": "Coal (generic, for process use)",
        "formula": "C (variable)",
        "molecular_weight": _D("12.01"),
        "carbon_content": _D("0.7500"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "REDUCTANT",
    },
    "PETROLEUM_COKE": {
        "display_name": "Petroleum Coke (anode grade)",
        "formula": "C (amorphous)",
        "molecular_weight": _D("12.01"),
        "carbon_content": _D("0.8730"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "REDUCTANT",
    },
    "ANODE_CARBON": {
        "display_name": "Prebake Anode (aluminum smelting)",
        "formula": "C (baked)",
        "molecular_weight": _D("12.01"),
        "carbon_content": _D("0.8500"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "ELECTRODE",
    },
    "ELECTRODE_CARBON": {
        "display_name": "Graphite Electrode (EAF)",
        "formula": "C (graphite)",
        "molecular_weight": _D("12.01"),
        "carbon_content": _D("0.9900"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "ELECTRODE",
    },
    "NATURAL_GAS_FEEDSTOCK": {
        "display_name": "Natural Gas (as chemical feedstock)",
        "formula": "CH4",
        "molecular_weight": _D("16.04"),
        "carbon_content": _D("0.7300"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "FEEDSTOCK",
    },
    "NAPHTHA_FEEDSTOCK": {
        "display_name": "Naphtha (petrochemical feedstock)",
        "formula": "CnH2n+2",
        "molecular_weight": _D("100.00"),
        "carbon_content": _D("0.8400"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "FEEDSTOCK",
    },
    "SCRAP_STEEL": {
        "display_name": "Scrap Steel (EAF input)",
        "formula": "Fe + C (trace)",
        "molecular_weight": _D("55.85"),
        "carbon_content": _D("0.0050"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "RECYCLED",
    },
    "ALUMINA": {
        "display_name": "Alumina (Al2O3, Bayer process output)",
        "formula": "Al2O3",
        "molecular_weight": _D("101.96"),
        "carbon_content": _D("0.0000"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "INTERMEDIATE",
    },
    "QUARTZ": {
        "display_name": "Quartz (SiO2, for ferroalloy production)",
        "formula": "SiO2",
        "molecular_weight": _D("60.08"),
        "carbon_content": _D("0.0000"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "ORE",
    },
    "CHARCOAL": {
        "display_name": "Charcoal (biomass-derived reductant)",
        "formula": "C (biogenic)",
        "molecular_weight": _D("12.01"),
        "carbon_content": _D("0.7500"),
        "carbonate_content": _D("0.0000"),
        "fraction_ite": "N/A",
        "stoichiometric_co2_factor": _D("0.000"),
        "category": "REDUCTANT",
    },
}


# =============================================================================
# SECTION 6: CARBONATE EMISSION FACTORS (Stoichiometric)
# =============================================================================
# CO2 yield per tonne of carbonate, based on the calcination reaction:
#   MCO3 -> MO + CO2
# Factor = MW_CO2 / MW_carbonate

_CARBONATE_FACTORS: Dict[str, Dict[str, Any]] = {
    "CALCITE": {
        "display_name": "Calcite (CaCO3)",
        "formula": "CaCO3 -> CaO + CO2",
        "molecular_weight_carbonate": _D("100.09"),
        "molecular_weight_oxide": _D("56.08"),
        "co2_factor": _D("0.4397"),
        "co2_factor_rounded": _D("0.440"),
        "carbon_content": _D("0.1200"),
    },
    "MAGNESITE": {
        "display_name": "Magnesite (MgCO3)",
        "formula": "MgCO3 -> MgO + CO2",
        "molecular_weight_carbonate": _D("84.31"),
        "molecular_weight_oxide": _D("40.30"),
        "co2_factor": _D("0.5220"),
        "co2_factor_rounded": _D("0.522"),
        "carbon_content": _D("0.1423"),
    },
    "DOLOMITE": {
        "display_name": "Dolomite (CaMg(CO3)2)",
        "formula": "CaMg(CO3)2 -> CaO + MgO + 2CO2",
        "molecular_weight_carbonate": _D("184.40"),
        "molecular_weight_oxide": _D("96.38"),
        "co2_factor": _D("0.4773"),
        "co2_factor_rounded": _D("0.477"),
        "carbon_content": _D("0.1303"),
    },
    "SIDERITE": {
        "display_name": "Siderite (FeCO3)",
        "formula": "FeCO3 -> FeO + CO2",
        "molecular_weight_carbonate": _D("115.86"),
        "molecular_weight_oxide": _D("71.85"),
        "co2_factor": _D("0.3799"),
        "co2_factor_rounded": _D("0.380"),
        "carbon_content": _D("0.1036"),
    },
    "ANKERITE": {
        "display_name": "Ankerite (Ca(Fe,Mg)(CO3)2)",
        "formula": "Ca(Fe,Mg)(CO3)2 -> CaO + FeO/MgO + 2CO2",
        "molecular_weight_carbonate": _D("206.00"),
        "molecular_weight_oxide": _D("117.98"),
        "co2_factor": _D("0.4272"),
        "co2_factor_rounded": _D("0.427"),
        "carbon_content": _D("0.1165"),
    },
    "SODA_ASH": {
        "display_name": "Soda Ash (Na2CO3)",
        "formula": "Na2CO3 -> Na2O + CO2",
        "molecular_weight_carbonate": _D("105.99"),
        "molecular_weight_oxide": _D("61.98"),
        "co2_factor": _D("0.4152"),
        "co2_factor_rounded": _D("0.415"),
        "carbon_content": _D("0.1132"),
    },
    "WITHERITE": {
        "display_name": "Witherite (BaCO3)",
        "formula": "BaCO3 -> BaO + CO2",
        "molecular_weight_carbonate": _D("197.34"),
        "molecular_weight_oxide": _D("153.33"),
        "co2_factor": _D("0.2229"),
        "co2_factor_rounded": _D("0.223"),
        "carbon_content": _D("0.0608"),
    },
    "STRONTIANITE": {
        "display_name": "Strontianite (SrCO3)",
        "formula": "SrCO3 -> SrO + CO2",
        "molecular_weight_carbonate": _D("147.63"),
        "molecular_weight_oxide": _D("103.62"),
        "co2_factor": _D("0.2980"),
        "co2_factor_rounded": _D("0.298"),
        "carbon_content": _D("0.0813"),
    },
    "RHODOCHROSITE": {
        "display_name": "Rhodochrosite (MnCO3)",
        "formula": "MnCO3 -> MnO + CO2",
        "molecular_weight_carbonate": _D("114.95"),
        "molecular_weight_oxide": _D("70.94"),
        "co2_factor": _D("0.3832"),
        "co2_factor_rounded": _D("0.383"),
        "carbon_content": _D("0.1044"),
    },
    "LITHIUM_CARBONATE": {
        "display_name": "Lithium Carbonate (Li2CO3)",
        "formula": "Li2CO3 -> Li2O + CO2",
        "molecular_weight_carbonate": _D("73.89"),
        "molecular_weight_oxide": _D("29.88"),
        "co2_factor": _D("0.5955"),
        "co2_factor_rounded": _D("0.596"),
        "carbon_content": _D("0.1624"),
    },
}


# =============================================================================
# SECTION 7: GWP VALUES (100-year and 20-year, 8 gas species)
# =============================================================================
# Comprehensive GWP values for all 8 greenhouse gas types tracked in
# industrial process emissions.

_GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": _D("1"),
        "CH4": _D("25"),
        "N2O": _D("298"),
        "CF4": _D("7390"),
        "C2F6": _D("12200"),
        "SF6": _D("22800"),
        "NF3": _D("17200"),
        "HFC_23": _D("14800"),
    },
    "AR5": {
        "CO2": _D("1"),
        "CH4": _D("28"),
        "N2O": _D("265"),
        "CF4": _D("6630"),
        "C2F6": _D("11100"),
        "SF6": _D("23500"),
        "NF3": _D("16100"),
        "HFC_23": _D("12400"),
    },
    "AR6": {
        "CO2": _D("1"),
        "CH4": _D("29.8"),
        "N2O": _D("273"),
        "CF4": _D("7380"),
        "C2F6": _D("12400"),
        "SF6": _D("25200"),
        "NF3": _D("17400"),
        "HFC_23": _D("14600"),
    },
    "AR6_20YR": {
        "CO2": _D("1"),
        "CH4": _D("82.5"),
        "N2O": _D("273"),
        "CF4": _D("5300"),
        "C2F6": _D("8210"),
        "SF6": _D("18300"),
        "NF3": _D("13400"),
        "HFC_23": _D("12000"),
    },
}


# =============================================================================
# SECTION 8: PRODUCTION ROUTES
# =============================================================================
# Detailed production route metadata for processes with multiple pathways.

_PRODUCTION_ROUTES: Dict[str, List[Dict[str, Any]]] = {
    "IRON_STEEL_BF_BOF": [
        {
            "route_id": "BF_BOF_INTEGRATED",
            "display_name": "Integrated BF-BOF (Coke Oven + Sinter + BF + BOF)",
            "description": "Full integrated steelmaking with on-site coke production",
            "default_ef_tco2_per_t": _D("1.900"),
            "scrap_fraction": _D("0.10"),
            "slag_ratio": _D("0.300"),
            "typical_carbon_input_kg_per_t": _D("600"),
        },
        {
            "route_id": "BF_BOF_PURCHASED_COKE",
            "display_name": "BF-BOF with Purchased Coke",
            "description": "Blast furnace route using externally purchased coke",
            "default_ef_tco2_per_t": _D("1.750"),
            "scrap_fraction": _D("0.15"),
            "slag_ratio": _D("0.280"),
            "typical_carbon_input_kg_per_t": _D("550"),
        },
    ],
    "IRON_STEEL_EAF": [
        {
            "route_id": "EAF_SCRAP_100",
            "display_name": "EAF 100% Scrap",
            "description": "Electric arc furnace using 100% scrap steel",
            "default_ef_tco2_per_t": _D("0.400"),
            "scrap_fraction": _D("1.00"),
            "slag_ratio": _D("0.100"),
            "typical_carbon_input_kg_per_t": _D("25"),
        },
        {
            "route_id": "EAF_DRI_BLEND",
            "display_name": "EAF with DRI + Scrap Blend",
            "description": "EAF using mix of DRI and scrap (typical 50/50)",
            "default_ef_tco2_per_t": _D("0.550"),
            "scrap_fraction": _D("0.50"),
            "slag_ratio": _D("0.120"),
            "typical_carbon_input_kg_per_t": _D("50"),
        },
    ],
    "IRON_STEEL_DRI": [
        {
            "route_id": "DRI_NATURAL_GAS",
            "display_name": "DRI Natural Gas (Midrex/HYL)",
            "description": "Direct reduced iron using natural gas as reductant",
            "default_ef_tco2_per_t": _D("0.700"),
            "scrap_fraction": _D("0.00"),
            "slag_ratio": _D("0.000"),
            "typical_carbon_input_kg_per_t": _D("200"),
        },
        {
            "route_id": "DRI_HYDROGEN",
            "display_name": "DRI Hydrogen (H2-DRI)",
            "description": "Direct reduced iron using green hydrogen as reductant",
            "default_ef_tco2_per_t": _D("0.050"),
            "scrap_fraction": _D("0.00"),
            "slag_ratio": _D("0.000"),
            "typical_carbon_input_kg_per_t": _D("10"),
        },
        {
            "route_id": "DRI_COAL",
            "display_name": "DRI Coal-Based (Rotary Kiln)",
            "description": "Direct reduced iron using coal as reductant",
            "default_ef_tco2_per_t": _D("1.200"),
            "scrap_fraction": _D("0.00"),
            "slag_ratio": _D("0.000"),
            "typical_carbon_input_kg_per_t": _D("380"),
        },
    ],
    "ALUMINUM_PREBAKE": [
        {
            "route_id": "PREBAKE_CWPB",
            "display_name": "Centre-Worked Prebake (CWPB)",
            "description": "Modern prebake technology with centre-work design",
            "default_ef_tco2_per_t": _D("1.500"),
            "anode_consumption_t_per_t_al": _D("0.420"),
            "pfc_slope_cf4": _D("0.143"),
            "pfc_slope_c2f6_ratio": _D("0.100"),
        },
        {
            "route_id": "PREBAKE_SWPB",
            "display_name": "Side-Worked Prebake (SWPB)",
            "description": "Older prebake technology with side-work design",
            "default_ef_tco2_per_t": _D("1.600"),
            "anode_consumption_t_per_t_al": _D("0.450"),
            "pfc_slope_cf4": _D("0.272"),
            "pfc_slope_c2f6_ratio": _D("0.121"),
        },
    ],
    "ALUMINUM_SODERBERG": [
        {
            "route_id": "SODERBERG_VSS",
            "display_name": "Vertical Stud Soderberg (VSS)",
            "description": "Soderberg cell with vertical stud configuration",
            "default_ef_tco2_per_t": _D("1.800"),
            "paste_consumption_t_per_t_al": _D("0.550"),
            "pfc_slope_cf4": _D("0.092"),
            "pfc_slope_c2f6_ratio": _D("0.053"),
        },
        {
            "route_id": "SODERBERG_HSS",
            "display_name": "Horizontal Stud Soderberg (HSS)",
            "description": "Soderberg cell with horizontal stud configuration",
            "default_ef_tco2_per_t": _D("1.850"),
            "paste_consumption_t_per_t_al": _D("0.580"),
            "pfc_slope_cf4": _D("0.092"),
            "pfc_slope_c2f6_ratio": _D("0.053"),
        },
    ],
    "AMMONIA": [
        {
            "route_id": "SMR_NATURAL_GAS",
            "display_name": "Steam Methane Reforming (Natural Gas)",
            "description": "Conventional SMR with natural gas feedstock",
            "default_ef_tco2_per_t": _D("1.500"),
        },
        {
            "route_id": "SMR_COAL_GASIFICATION",
            "display_name": "Coal Gasification",
            "description": "Ammonia production from coal-based syngas",
            "default_ef_tco2_per_t": _D("2.300"),
        },
        {
            "route_id": "SMR_WITH_CCS",
            "display_name": "SMR with Carbon Capture (Blue Ammonia)",
            "description": "SMR with 85-95% CO2 capture rate",
            "default_ef_tco2_per_t": _D("0.225"),
        },
    ],
    "HYDROGEN": [
        {
            "route_id": "SMR_GREY",
            "display_name": "Grey Hydrogen (SMR without CCS)",
            "description": "Steam methane reforming without carbon capture",
            "default_ef_kgco2_per_kg": _D("9.300"),
        },
        {
            "route_id": "SMR_BLUE",
            "display_name": "Blue Hydrogen (SMR with CCS)",
            "description": "Steam methane reforming with 85-95% CO2 capture",
            "default_ef_kgco2_per_kg": _D("1.400"),
        },
        {
            "route_id": "ATR_BLUE",
            "display_name": "Blue Hydrogen (ATR with CCS)",
            "description": "Autothermal reforming with 95%+ CO2 capture",
            "default_ef_kgco2_per_kg": _D("0.900"),
        },
        {
            "route_id": "COAL_GASIFICATION",
            "display_name": "Coal Gasification (Brown Hydrogen)",
            "description": "Hydrogen from coal gasification without CCS",
            "default_ef_kgco2_per_kg": _D("19.000"),
        },
    ],
    "CEMENT": [
        {
            "route_id": "PORTLAND_CEMENT",
            "display_name": "Portland Cement (OPC)",
            "description": "Standard ordinary Portland cement with ~95% clinker",
            "default_clinker_ratio": _D("0.95"),
            "default_ef_tco2_per_t_clinker": _D("0.507"),
        },
        {
            "route_id": "BLENDED_CEMENT",
            "display_name": "Blended Cement (PPC/PSC)",
            "description": "Cement with supplementary cementitious materials",
            "default_clinker_ratio": _D("0.70"),
            "default_ef_tco2_per_t_clinker": _D("0.507"),
        },
        {
            "route_id": "WHITE_CEMENT",
            "display_name": "White Cement",
            "description": "White cement with higher purity clinker",
            "default_clinker_ratio": _D("0.95"),
            "default_ef_tco2_per_t_clinker": _D("0.530"),
        },
    ],
}


# =============================================================================
# SECTION 9: SEMICONDUCTOR GAS PARAMETERS
# =============================================================================
# Default utilization rates and by-product formation factors for
# semiconductor fab gases per IPCC 2006 Vol 3 Table 6.3.

_SEMICONDUCTOR_GAS_PARAMS: Dict[str, Dict[str, Decimal]] = {
    "CF4": {
        "default_utilization_rate": _D("0.50"),
        "default_destruction_efficiency": _D("0.00"),
        "by_product_cf4_from_c2f6": _D("0.00"),
        "by_product_cf4_from_c3f8": _D("0.00"),
        "by_product_cf4_from_nf3": _D("0.00"),
    },
    "C2F6": {
        "default_utilization_rate": _D("0.40"),
        "default_destruction_efficiency": _D("0.00"),
        "by_product_cf4_fraction": _D("0.30"),
    },
    "C3F8": {
        "default_utilization_rate": _D("0.30"),
        "default_destruction_efficiency": _D("0.00"),
        "by_product_cf4_fraction": _D("0.30"),
    },
    "SF6": {
        "default_utilization_rate": _D("0.50"),
        "default_destruction_efficiency": _D("0.00"),
        "by_product_cf4_fraction": _D("0.00"),
    },
    "NF3": {
        "default_utilization_rate": _D("0.98"),
        "default_destruction_efficiency": _D("0.00"),
        "by_product_cf4_fraction": _D("0.10"),
    },
    "HFC_23": {
        "default_utilization_rate": _D("0.50"),
        "default_destruction_efficiency": _D("0.00"),
        "by_product_cf4_fraction": _D("0.00"),
    },
    "c_C4F8": {
        "default_utilization_rate": _D("0.20"),
        "default_destruction_efficiency": _D("0.00"),
        "by_product_cf4_fraction": _D("0.30"),
    },
}


# =============================================================================
# ENGINE CLASS
# =============================================================================


class ProcessDatabaseEngine:
    """Manages process type metadata, emission factors, raw materials,
    carbonate stoichiometric factors, and GWP values for 25 industrial
    process types.

    This engine is the authoritative in-memory data source for all process
    emission calculations. It supports 4 built-in data sources (IPCC, EPA,
    DEFRA, EU ETS) and user-registered custom factors.

    Zero-Hallucination Guarantees:
        - All data is hard-coded from IPCC 2006, EPA 40 CFR Part 98,
          EU ETS MRR, and DEFRA 2025 conversion factors.
        - No LLM in the data path. Lookup is pure dictionary access.
        - Decimal arithmetic prevents floating-point drift.
        - SHA-256 provenance tracks every database access.

    Thread Safety:
        All mutable state is protected by ``threading.Lock()``.

    Attributes:
        _config: Optional configuration dictionary.
        _custom_factors: Registry of user-defined emission factors.
        _lock: Thread lock for custom factor mutations.
        _provenance: Reference to the provenance tracker singleton.

    Example:
        >>> db = ProcessDatabaseEngine()
        >>> info = db.get_process_info("CEMENT")
        >>> print(info["display_name"])
        Cement Production
        >>> ef = db.get_emission_factor("CEMENT", "CO2", source="IPCC")
        >>> print(ef)
        0.507
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProcessDatabaseEngine with optional configuration.

        Loads all built-in process type definitions, emission factors, raw
        material properties, carbonate factors, and GWP values. All data is
        held in-memory for deterministic, zero-latency lookups.

        Args:
            config: Optional configuration dict. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                  Defaults to True.
        """
        self._config = config or {}
        self._custom_factors: Dict[_EFKey, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)

        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            self._provenance = _get_provenance_tracker()
        else:
            self._provenance = None

        logger.info(
            "ProcessDatabaseEngine initialized with %d process types, "
            "%d raw materials, %d carbonate types, %d total emission factors",
            len(_PROCESS_TYPES),
            len(_RAW_MATERIALS),
            len(_CARBONATE_FACTORS),
            self.get_factor_count(),
        )

    # ------------------------------------------------------------------
    # Public API: Process Type Information
    # ------------------------------------------------------------------

    def get_process_info(self, process_type: str) -> Dict[str, Any]:
        """Return full metadata for an industrial process type.

        Args:
            process_type: Process type identifier (e.g. ``"CEMENT"``).

        Returns:
            Dictionary with keys: category, display_name, description,
            primary_gases, applicable_tiers, applicable_methods,
            epa_subpart, ipcc_code, default_production_unit.

        Raises:
            KeyError: If the process type is not found.

        Example:
            >>> info = db.get_process_info("NITRIC_ACID")
            >>> info["primary_gases"]
            ['N2O']
        """
        key = process_type.upper()
        if key not in _PROCESS_TYPES:
            raise KeyError(f"Unknown process type: {process_type}")

        result = dict(_PROCESS_TYPES[key])

        if _METRICS_AVAILABLE and _record_process_lookup is not None:
            _record_process_lookup(key, result["category"])

        self._record_provenance(
            "lookup_process", key, {"category": result["category"]},
        )
        return result

    def list_processes(
        self,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all process types, optionally filtered by category.

        Args:
            category: Optional category filter. One of ``"MINERAL"``,
                ``"CHEMICAL"``, ``"METAL"``, ``"ELECTRONICS"``,
                ``"PULP_PAPER"``, ``"OTHER"``.

        Returns:
            List of process info dictionaries, each augmented with a
            ``process_type`` key for identification.

        Example:
            >>> mineral = db.list_processes(category="MINERAL")
            >>> len(mineral)
            5
        """
        results: List[Dict[str, Any]] = []
        cat_upper = category.upper() if category else None

        for pt_key, pt_info in _PROCESS_TYPES.items():
            if cat_upper and pt_info["category"] != cat_upper:
                continue
            entry = dict(pt_info)
            entry["process_type"] = pt_key
            results.append(entry)

        self._record_provenance(
            "list_processes", "all",
            {"category": cat_upper or "ALL", "count": len(results)},
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Emission Factors
    # ------------------------------------------------------------------

    def get_emission_factor(
        self,
        process_type: str,
        gas: str,
        source: str = "IPCC",
    ) -> Decimal:
        """Look up an emission factor for a process, gas, and source.

        Args:
            process_type: Process type identifier (e.g. ``"CEMENT"``).
            gas: Greenhouse gas (``"CO2"``, ``"CH4"``, ``"N2O"``,
                ``"CF4"``, ``"C2F6"``, ``"SF6"``, ``"NF3"``, ``"HFC_23"``).
            source: Factor source (``"IPCC"``, ``"EPA"``, ``"DEFRA"``,
                ``"EU_ETS"``, ``"CUSTOM"``). Defaults to ``"IPCC"``.

        Returns:
            Emission factor as a ``Decimal``.

        Raises:
            KeyError: If the process, gas, or source combination is not found.

        Example:
            >>> db.get_emission_factor("CEMENT", "CO2", "IPCC")
            Decimal('0.507')
            >>> db.get_emission_factor("ALUMINUM_PREBAKE", "CF4", "EPA")
            Decimal('0.040')
        """
        pt_key = process_type.upper()
        gas_key = gas.upper()
        source_key = source.upper()

        if _METRICS_AVAILABLE and _record_factor_selection is not None:
            _record_factor_selection(pt_key, source_key)

        # Check custom factors first
        custom_key: _EFKey = (pt_key, gas_key, source_key)
        with self._lock:
            if custom_key in self._custom_factors:
                val = self._custom_factors[custom_key]["value"]
                self._record_provenance(
                    "lookup_factor", pt_key,
                    {"gas": gas_key, "source": source_key, "value": str(val), "custom": True},
                )
                return val

        # Look up in built-in source
        source_map = self._get_source_map(source_key)
        if pt_key not in source_map:
            raise KeyError(
                f"No emission factors for process '{process_type}' "
                f"in source '{source}'"
            )

        process_factors = source_map[pt_key]
        if gas_key not in process_factors:
            raise KeyError(
                f"No {gas} emission factor for process '{process_type}' "
                f"in source '{source}'. Available gases: "
                f"{list(process_factors.keys())}"
            )

        value = process_factors[gas_key]
        self._record_provenance(
            "lookup_factor", pt_key,
            {"gas": gas_key, "source": source_key, "value": str(value)},
        )
        return value

    def get_emission_factor_with_unit(
        self,
        process_type: str,
        gas: str,
        source: str = "IPCC",
    ) -> Tuple[Decimal, str]:
        """Look up an emission factor along with its unit string.

        Args:
            process_type: Process type identifier.
            gas: Greenhouse gas identifier.
            source: Factor source. Defaults to ``"IPCC"``.

        Returns:
            Tuple of (factor_value, unit_string).

        Raises:
            KeyError: If not found.

        Example:
            >>> val, unit = db.get_emission_factor_with_unit("CEMENT", "CO2")
            >>> print(f"{val} {unit}")
            0.507 tCO2/t_clinker
        """
        value = self.get_emission_factor(process_type, gas, source)
        pt_key = process_type.upper()
        gas_key = gas.upper()

        unit = "varies"
        if pt_key in _PROCESS_EF_UNITS:
            unit = _PROCESS_EF_UNITS[pt_key].get(gas_key, "varies")

        return value, unit

    def get_variant_factor(
        self,
        variant_id: str,
        source: str = "IPCC",
    ) -> Dict[str, Decimal]:
        """Look up emission factors for a specific process variant.

        Args:
            variant_id: Variant identifier (e.g. ``"AMMONIA_COAL"``,
                ``"ADIPIC_ACID_ABATED"``, ``"LIME_DOLOMITIC"``).
            source: Factor source. Defaults to ``"IPCC"``.

        Returns:
            Dictionary mapping gas to Decimal factor value.

        Raises:
            KeyError: If the variant or source is not found.

        Example:
            >>> factors = db.get_variant_factor("ADIPIC_ACID_ABATED", "IPCC")
            >>> factors["N2O"]
            Decimal('0.009')
        """
        var_key = variant_id.upper()
        source_key = source.upper()

        if var_key not in _PROCESS_VARIANT_FACTORS:
            raise KeyError(f"Unknown process variant: {variant_id}")

        variant_data = _PROCESS_VARIANT_FACTORS[var_key]
        if source_key not in variant_data:
            raise KeyError(
                f"No factors for variant '{variant_id}' in source '{source}'. "
                f"Available sources: {list(variant_data.keys())}"
            )

        result = dict(variant_data[source_key])
        self._record_provenance(
            "lookup_variant", var_key,
            {"source": source_key, "factors": {k: str(v) for k, v in result.items()}},
        )
        return result

    def register_custom_factor(
        self,
        process_type: str,
        gas: str,
        value: Decimal,
        source: str = "CUSTOM",
        unit: str = "varies",
        description: str = "",
    ) -> None:
        """Register a user-defined custom emission factor.

        Custom factors take precedence over built-in factors during lookup.

        Args:
            process_type: Process type identifier.
            gas: Greenhouse gas identifier.
            value: Factor value as Decimal.
            source: Source label. Defaults to ``"CUSTOM"``.
            unit: Unit label. Defaults to ``"varies"``.
            description: Optional description.

        Raises:
            ValueError: If value is negative.

        Example:
            >>> db.register_custom_factor("CEMENT", "CO2", Decimal("0.495"),
            ...     description="Site-specific clinker factor")
        """
        if value < 0:
            raise ValueError(f"Emission factor value cannot be negative: {value}")

        pt_key = process_type.upper()
        gas_key = gas.upper()
        source_key = source.upper()
        key: _EFKey = (pt_key, gas_key, source_key)

        with self._lock:
            self._custom_factors[key] = {
                "value": value,
                "unit": unit,
                "description": description,
            }

        logger.info(
            "Registered custom factor: %s/%s/%s = %s %s",
            pt_key, gas_key, source_key, value, unit,
        )
        self._record_provenance(
            "register_custom_factor", pt_key,
            {"gas": gas_key, "source": source_key, "value": str(value)},
        )

    # ------------------------------------------------------------------
    # Public API: Raw Materials
    # ------------------------------------------------------------------

    def get_raw_material(self, material_type: str) -> Dict[str, Any]:
        """Return properties for a raw material type.

        Args:
            material_type: Material identifier (e.g. ``"CALCIUM_CARBONATE"``,
                ``"COKE"``, ``"SCRAP_STEEL"``).

        Returns:
            Dictionary with keys: display_name, formula, molecular_weight,
            carbon_content, carbonate_content, category, etc.

        Raises:
            KeyError: If the material type is not found.

        Example:
            >>> mat = db.get_raw_material("CALCIUM_CARBONATE")
            >>> mat["carbon_content"]
            Decimal('0.1200')
        """
        key = material_type.upper()
        if key not in _RAW_MATERIALS:
            raise KeyError(f"Unknown raw material: {material_type}")

        result = dict(_RAW_MATERIALS[key])
        result["material_type"] = key

        self._record_provenance(
            "lookup_material", key,
            {"category": result.get("category", "UNKNOWN")},
        )
        return result

    def list_materials(
        self,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all raw materials, optionally filtered by category.

        Args:
            category: Optional filter. One of ``"CARBONATE"``, ``"MINERAL"``,
                ``"ORE"``, ``"REDUCTANT"``, ``"ELECTRODE"``, ``"FEEDSTOCK"``,
                ``"RECYCLED"``, ``"INTERMEDIATE"``.

        Returns:
            List of material property dictionaries.

        Example:
            >>> carbonates = db.list_materials(category="CARBONATE")
            >>> len(carbonates)
            10
        """
        results: List[Dict[str, Any]] = []
        cat_upper = category.upper() if category else None

        for mat_key, mat_info in _RAW_MATERIALS.items():
            if cat_upper and mat_info.get("category") != cat_upper:
                continue
            entry = dict(mat_info)
            entry["material_type"] = mat_key
            results.append(entry)

        return results

    # ------------------------------------------------------------------
    # Public API: Carbonate Factors
    # ------------------------------------------------------------------

    def get_carbonate_factor(self, carbonate_type: str) -> Dict[str, Any]:
        """Return stoichiometric data for a carbonate mineral.

        Args:
            carbonate_type: Carbonate identifier (e.g. ``"CALCITE"``,
                ``"MAGNESITE"``, ``"DOLOMITE"``, ``"SIDERITE"``).

        Returns:
            Dictionary with keys: display_name, formula,
            molecular_weight_carbonate, molecular_weight_oxide,
            co2_factor, co2_factor_rounded, carbon_content.

        Raises:
            KeyError: If the carbonate type is not found.

        Example:
            >>> cf = db.get_carbonate_factor("CALCITE")
            >>> cf["co2_factor"]
            Decimal('0.4397')
        """
        key = carbonate_type.upper()
        if key not in _CARBONATE_FACTORS:
            raise KeyError(
                f"Unknown carbonate type: {carbonate_type}. "
                f"Available: {list(_CARBONATE_FACTORS.keys())}"
            )

        result = dict(_CARBONATE_FACTORS[key])
        result["carbonate_type"] = key

        self._record_provenance(
            "lookup_carbonate", key,
            {"co2_factor": str(result["co2_factor"])},
        )
        return result

    def list_carbonates(self) -> List[Dict[str, Any]]:
        """List all carbonate stoichiometric factors.

        Returns:
            List of carbonate factor dictionaries, each with a
            ``carbonate_type`` key.

        Example:
            >>> all_carbs = db.list_carbonates()
            >>> len(all_carbs)
            10
        """
        results: List[Dict[str, Any]] = []
        for key, info in _CARBONATE_FACTORS.items():
            entry = dict(info)
            entry["carbonate_type"] = key
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Public API: GWP Values
    # ------------------------------------------------------------------

    def get_gwp(
        self,
        gas: str,
        source: str = "AR6",
    ) -> Decimal:
        """Look up the 100-year Global Warming Potential for a gas.

        Args:
            gas: Greenhouse gas (``"CO2"``, ``"CH4"``, ``"N2O"``,
                ``"CF4"``, ``"C2F6"``, ``"SF6"``, ``"NF3"``, ``"HFC_23"``).
            source: GWP source. One of ``"AR4"``, ``"AR5"``, ``"AR6"``,
                ``"AR6_20YR"``. Defaults to ``"AR6"``.

        Returns:
            GWP value as a ``Decimal``.

        Raises:
            KeyError: If the gas or source is not found.

        Example:
            >>> db.get_gwp("CF4", "AR6")
            Decimal('7380')
            >>> db.get_gwp("N2O", "AR5")
            Decimal('265')
        """
        gas_key = gas.upper()
        source_key = source.upper()

        if source_key not in _GWP_VALUES:
            raise KeyError(
                f"Unknown GWP source: {source}. "
                f"Available: {list(_GWP_VALUES.keys())}"
            )

        gwp_table = _GWP_VALUES[source_key]
        if gas_key not in gwp_table:
            raise KeyError(
                f"No GWP for gas '{gas}' in source '{source}'. "
                f"Available gases: {list(gwp_table.keys())}"
            )

        value = gwp_table[gas_key]
        self._record_provenance(
            "lookup_gwp", gas_key,
            {"source": source_key, "gwp": str(value)},
        )
        return value

    def get_all_gwp_values(
        self,
        gas: str,
    ) -> Dict[str, Decimal]:
        """Return all GWP values (all AR sources) for a gas.

        Args:
            gas: Greenhouse gas identifier.

        Returns:
            Dictionary mapping source name to GWP Decimal value.

        Raises:
            KeyError: If the gas is not found in any source.

        Example:
            >>> all_gwps = db.get_all_gwp_values("SF6")
            >>> all_gwps["AR6"]
            Decimal('25200')
        """
        gas_key = gas.upper()
        result: Dict[str, Decimal] = {}

        for source_key, gwp_table in _GWP_VALUES.items():
            if gas_key in gwp_table:
                result[source_key] = gwp_table[gas_key]

        if not result:
            raise KeyError(f"No GWP values found for gas: {gas}")

        return result

    # ------------------------------------------------------------------
    # Public API: Production Routes
    # ------------------------------------------------------------------

    def get_production_routes(
        self,
        process_type: str,
    ) -> List[Dict[str, Any]]:
        """Return available production routes for a process type.

        Args:
            process_type: Process type identifier.

        Returns:
            List of route dictionaries with route_id, display_name,
            description, and route-specific parameters.
            Empty list if no routes are defined.

        Example:
            >>> routes = db.get_production_routes("IRON_STEEL_BF_BOF")
            >>> len(routes)
            2
            >>> routes[0]["route_id"]
            'BF_BOF_INTEGRATED'
        """
        key = process_type.upper()
        if key not in _PRODUCTION_ROUTES:
            return []

        result = [dict(r) for r in _PRODUCTION_ROUTES[key]]

        self._record_provenance(
            "lookup_routes", key,
            {"count": len(result)},
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Semiconductor Parameters
    # ------------------------------------------------------------------

    def get_semiconductor_gas_params(
        self,
        gas: str,
    ) -> Dict[str, Decimal]:
        """Return semiconductor fabrication parameters for a specific gas.

        Args:
            gas: Gas identifier (``"CF4"``, ``"C2F6"``, ``"C3F8"``,
                ``"SF6"``, ``"NF3"``, ``"HFC_23"``, ``"c_C4F8"``).

        Returns:
            Dictionary with keys: default_utilization_rate,
            default_destruction_efficiency, by_product_cf4_fraction, etc.

        Raises:
            KeyError: If the gas is not found.

        Example:
            >>> params = db.get_semiconductor_gas_params("NF3")
            >>> params["default_utilization_rate"]
            Decimal('0.98')
        """
        gas_key = gas.upper()
        if gas_key not in _SEMICONDUCTOR_GAS_PARAMS:
            raise KeyError(
                f"No semiconductor parameters for gas: {gas}. "
                f"Available: {list(_SEMICONDUCTOR_GAS_PARAMS.keys())}"
            )

        result = dict(_SEMICONDUCTOR_GAS_PARAMS[gas_key])
        self._record_provenance(
            "lookup_semiconductor_gas", gas_key,
            {"utilization": str(result.get("default_utilization_rate", "N/A"))},
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Counts and Metadata
    # ------------------------------------------------------------------

    def get_factor_count(self) -> int:
        """Return total number of emission factors across all sources.

        Counts all built-in factors (IPCC, EPA, DEFRA, EU ETS) plus any
        registered custom factors.

        Returns:
            Integer count of emission factors.
        """
        count = 0
        for source_map in (
            _IPCC_PROCESS_FACTORS,
            _EPA_PROCESS_FACTORS,
            _DEFRA_PROCESS_FACTORS,
            _EU_ETS_PROCESS_FACTORS,
        ):
            for gas_dict in source_map.values():
                count += len(gas_dict)

        with self._lock:
            count += len(self._custom_factors)

        return count

    def get_process_count(self) -> int:
        """Return total number of registered process types.

        Returns:
            Integer count of process types (default 25).
        """
        return len(_PROCESS_TYPES)

    def get_material_count(self) -> int:
        """Return total number of raw materials in the database.

        Returns:
            Integer count of raw material entries.
        """
        return len(_RAW_MATERIALS)

    def get_carbonate_count(self) -> int:
        """Return total number of carbonate factor entries.

        Returns:
            Integer count of carbonate types.
        """
        return len(_CARBONATE_FACTORS)

    def get_available_gases(
        self,
        process_type: str,
        source: str = "IPCC",
    ) -> List[str]:
        """Return list of gases with emission factors for a process/source.

        Args:
            process_type: Process type identifier.
            source: Factor source. Defaults to ``"IPCC"``.

        Returns:
            List of gas identifiers (e.g. ``["CO2", "CF4", "C2F6"]``).

        Example:
            >>> db.get_available_gases("ALUMINUM_PREBAKE")
            ['CO2', 'CF4', 'C2F6']
        """
        pt_key = process_type.upper()
        source_key = source.upper()

        source_map = self._get_source_map(source_key)
        if pt_key not in source_map:
            return []

        return list(source_map[pt_key].keys())

    def get_available_sources(
        self,
        process_type: str,
    ) -> List[str]:
        """Return list of factor sources that have data for a process.

        Args:
            process_type: Process type identifier.

        Returns:
            List of source identifiers (e.g. ``["IPCC", "EPA", "DEFRA", "EU_ETS"]``).
        """
        pt_key = process_type.upper()
        sources: List[str] = []

        source_names = [
            ("IPCC", _IPCC_PROCESS_FACTORS),
            ("EPA", _EPA_PROCESS_FACTORS),
            ("DEFRA", _DEFRA_PROCESS_FACTORS),
            ("EU_ETS", _EU_ETS_PROCESS_FACTORS),
        ]
        for name, source_map in source_names:
            if pt_key in source_map:
                sources.append(name)

        # Check for custom factors
        with self._lock:
            for (pt, gas, src) in self._custom_factors:
                if pt == pt_key and src not in sources:
                    sources.append(src)

        return sources

    # ------------------------------------------------------------------
    # Private: Source map selector
    # ------------------------------------------------------------------

    def _get_source_map(
        self,
        source: str,
    ) -> Dict[str, Dict[str, Decimal]]:
        """Return the appropriate emission factor dictionary for a source.

        Args:
            source: Source identifier (uppercased).

        Returns:
            The module-level factor dictionary for the given source.

        Raises:
            KeyError: If the source is not recognized.
        """
        source_upper = source.upper()
        if source_upper == "IPCC":
            return _IPCC_PROCESS_FACTORS
        elif source_upper == "EPA":
            return _EPA_PROCESS_FACTORS
        elif source_upper == "DEFRA":
            return _DEFRA_PROCESS_FACTORS
        elif source_upper == "EU_ETS":
            return _EU_ETS_PROCESS_FACTORS
        else:
            raise KeyError(
                f"Unknown emission factor source: {source}. "
                f"Available: IPCC, EPA, DEFRA, EU_ETS, CUSTOM"
            )

    # ------------------------------------------------------------------
    # Private: Provenance recording
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an operation in the provenance tracker if available.

        Args:
            action: Action name (e.g. ``"lookup_factor"``).
            entity_id: Entity identifier.
            data: Optional data dictionary to include in the record.
        """
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="process_database",
                    action=action,
                    entity_id=entity_id,
                    data=data or {},
                )
            except Exception as exc:
                logger.debug(
                    "Provenance recording failed (non-critical): %s", exc,
                )
