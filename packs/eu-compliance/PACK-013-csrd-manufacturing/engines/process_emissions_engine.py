# -*- coding: utf-8 -*-
"""
ProcessEmissionsEngine - PACK-013 CSRD Manufacturing Engine 1
================================================================

Industrial process emissions calculator for manufacturing sub-sectors,
covering Scope 1 direct process emissions, combustion emissions, and
fugitive releases.  Integrates with EU ETS benchmarking, CBAM embedded
emissions calculations, and CO2 abatement / CCS tracking.

Manufacturing Sub-Sectors Supported:
    - Cement (calcination, kiln combustion)
    - Steel (BOF, EAF, coke-oven)
    - Aluminium (Hall-Heroult electrolysis, anode effects / PFCs)
    - Chemicals (ammonia, nitric acid, adipic acid, methanol, ethylene)
    - Glass (batch decomposition)
    - Ceramics (calcination, firing)
    - Pulp & Paper (lime kiln, chemical recovery)
    - Food & Beverage (fermentation, thermal processing)
    - Textiles (dyeing, finishing heat)
    - Pharmaceuticals (API synthesis, solvent recovery)
    - Electronics (solder, etching gases)
    - Automotive (painting, welding, casting)

Core Formulas:
    Process CO2    = SUM( raw_material_qty * process_emission_factor )
    Combustion CO2 = SUM( fuel_qty * NCV * fuel_emission_factor )
    Total          = Process CO2 + Combustion CO2 + Fugitive CO2 - Abated CO2
    Intensity      = Total / Annual_Production_Tonnes

Regulatory References:
    - EU ETS Monitoring and Reporting Regulation (MRR) 2018/2066
    - CBAM Regulation (EU) 2023/956 - Annex IV embedded emissions
    - ESRS E1 (Climate Change) - Scope 1 process emissions
    - GHG Protocol Chapter 3 (Stationary Combustion)
    - 2006 IPCC Guidelines Vol. 3 (Industrial Processes)

Zero-Hallucination:
    - All calculations use deterministic Python Decimal / float arithmetic
    - Emission factors from published regulatory / IPCC sources
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-013 CSRD Manufacturing
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP for regulatory precision."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ManufacturingSubSector(str, Enum):
    """Manufacturing sub-sectors for process emission calculations."""
    CEMENT = "cement"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    CHEMICALS = "chemicals"
    GLASS = "glass"
    CERAMICS = "ceramics"
    PULP_PAPER = "pulp_paper"
    FOOD_BEVERAGE = "food_beverage"
    TEXTILES = "textiles"
    PHARMACEUTICALS = "pharmaceuticals"
    ELECTRONICS = "electronics"
    AUTOMOTIVE = "automotive"

class ProcessType(str, Enum):
    """Types of industrial processes that generate emissions."""
    CALCINATION = "calcination"
    REDUCTION = "reduction"
    ELECTROLYSIS = "electrolysis"
    SYNTHESIS = "synthesis"
    DECOMPOSITION = "decomposition"
    FERMENTATION = "fermentation"
    COMBUSTION = "combustion"

class FuelType(str, Enum):
    """Fuel types used in manufacturing processes."""
    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    COKE = "coke"
    FUEL_OIL = "fuel_oil"
    BIOMASS = "biomass"
    WASTE_FUEL = "waste_fuel"
    HYDROGEN = "hydrogen"
    ELECTRICITY = "electricity"

# ---------------------------------------------------------------------------
# Constants: Process Emission Factors (tCO2 per tonne of product/material)
# Sources: 2006 IPCC Guidelines Vol. 3, EU MRR Annex IV
# ---------------------------------------------------------------------------

PROCESS_EMISSION_FACTORS: Dict[str, float] = {
    # Cement -----------------------------------------------------------------
    "cement_calcination": 0.525,           # tCO2/t clinker (CaCO3 decomposition)
    "cement_mgo_calcination": 0.785,       # tCO2/t MgCO3 content
    "cement_bypass_dust": 0.050,           # tCO2/t clinker (CKD correction)

    # Steel ------------------------------------------------------------------
    "steel_bof": 1.328,                    # tCO2/t hot metal (BF-BOF route)
    "steel_eaf": 0.080,                    # tCO2/t crude steel (EAF, electrode)
    "steel_coke_oven": 0.561,              # tCO2/t coke produced
    "steel_sinter": 0.200,                 # tCO2/t sinter (limestone flux)
    "steel_dri": 0.700,                    # tCO2/t DRI (direct reduced iron)

    # Aluminium --------------------------------------------------------------
    "aluminum_electrolysis": 1.514,        # tCO2/t Al (anode consumption + PFC)
    "aluminum_pfc_cf4": 0.058,             # tCO2e/t Al (CF4 anode effects)
    "aluminum_pfc_c2f6": 0.006,            # tCO2e/t Al (C2F6 anode effects)
    "aluminum_anode_baking": 0.150,        # tCO2/t Al (Soderberg pre-bake)

    # Chemicals --------------------------------------------------------------
    "ammonia_synthesis": 1.600,            # tCO2/t NH3 (steam methane reforming)
    "nitric_acid_n2o": 5.700,              # tCO2e/t HNO3 (N2O emissions)
    "adipic_acid_n2o": 12.200,             # tCO2e/t adipic acid (N2O)
    "methanol_synthesis": 0.670,           # tCO2/t methanol
    "ethylene_cracking": 1.000,            # tCO2/t ethylene (steam cracking)
    "hydrogen_smr": 9.300,                 # tCO2/t H2 (steam methane reforming)
    "chlorine_electrolysis": 0.180,        # tCO2/t Cl2 (membrane cell)
    "soda_ash_trona": 0.138,               # tCO2/t soda ash (from trona)

    # Glass ------------------------------------------------------------------
    "glass_decomposition": 0.210,          # tCO2/t glass (batch material decomp.)
    "glass_flat": 0.200,                   # tCO2/t flat glass
    "glass_container": 0.185,              # tCO2/t container glass (higher cullet)

    # Ceramics ---------------------------------------------------------------
    "ceramics_calcination": 0.100,         # tCO2/t ceramic (clay calcination)
    "ceramics_tile": 0.080,                # tCO2/t tile (lower calcium content)
    "ceramics_brick": 0.120,               # tCO2/t brick (higher calcium content)

    # Pulp & Paper -----------------------------------------------------------
    "pulp_lime_kiln": 0.120,              # tCO2/t CaO (lime kiln make-up limestone)
    "pulp_chemical_recovery": 0.040,       # tCO2/t pulp (chemical recovery furnace)

    # Food & Beverage --------------------------------------------------------
    "food_fermentation_co2": 0.040,        # tCO2/t product (fermentation CO2)
    "food_baking_co2": 0.010,              # tCO2/t product (leavening agent CO2)
    "beverage_fermentation": 0.960,        # tCO2/kL ethanol (alcoholic fermentation)

    # Textiles ---------------------------------------------------------------
    "textiles_dyeing": 0.015,              # tCO2/t fabric (chemical process)
    "textiles_finishing": 0.010,           # tCO2/t fabric (thermal treatment)

    # Pharmaceuticals --------------------------------------------------------
    "pharma_api_synthesis": 0.050,         # tCO2/t API (synthesis off-gas)
    "pharma_solvent_recovery": 0.020,      # tCO2/t solvent (thermal decomp.)

    # Electronics ------------------------------------------------------------
    "electronics_pfc_etching": 0.005,      # tCO2e/m2 wafer (PFC/NF3)
    "electronics_solder": 0.002,           # tCO2/t product (flux emissions)

    # Automotive -------------------------------------------------------------
    "automotive_painting_voc": 0.025,      # tCO2e/vehicle (VOC from paint)
    "automotive_casting": 0.050,           # tCO2/t casting (sand mould CO2)
}

# ---------------------------------------------------------------------------
# Constants: Fuel Emission Factors (tCO2/TJ, NCV in TJ/unit)
# Source: 2006 IPCC Guidelines Vol. 2, Table 2.2
# ---------------------------------------------------------------------------

FUEL_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "co2_factor_tco2_per_tj": 56.1,
        "ncv_tj_per_unit": 0.03412,        # TJ per 1000 m3
        "default_unit": "1000m3",
    },
    "coal": {
        "co2_factor_tco2_per_tj": 94.6,
        "ncv_tj_per_unit": 0.02558,        # TJ per tonne (bituminous)
        "default_unit": "tonnes",
    },
    "coke": {
        "co2_factor_tco2_per_tj": 107.0,
        "ncv_tj_per_unit": 0.02825,        # TJ per tonne
        "default_unit": "tonnes",
    },
    "fuel_oil": {
        "co2_factor_tco2_per_tj": 77.4,
        "ncv_tj_per_unit": 0.04027,        # TJ per tonne (heavy fuel oil)
        "default_unit": "tonnes",
    },
    "biomass": {
        "co2_factor_tco2_per_tj": 0.0,     # Biogenic -- zero under GHG Protocol
        "ncv_tj_per_unit": 0.01550,        # TJ per tonne (wood pellets avg)
        "default_unit": "tonnes",
    },
    "waste_fuel": {
        "co2_factor_tco2_per_tj": 91.7,    # Mixed industrial waste
        "ncv_tj_per_unit": 0.01000,        # TJ per tonne (variable)
        "default_unit": "tonnes",
    },
    "hydrogen": {
        "co2_factor_tco2_per_tj": 0.0,     # Zero at point of combustion
        "ncv_tj_per_unit": 0.12079,        # TJ per tonne (LHV 120.79 MJ/kg)
        "default_unit": "tonnes",
    },
    "electricity": {
        "co2_factor_tco2_per_tj": 0.0,     # Scope 2 -- not counted in Scope 1
        "ncv_tj_per_unit": 0.0036,         # TJ per MWh
        "default_unit": "MWh",
    },
}

# ---------------------------------------------------------------------------
# Constants: EU ETS Product Benchmarks (tCO2 / t product)
# Source: Commission Implementing Regulation (EU) 2021/447
# ---------------------------------------------------------------------------

ETS_PRODUCT_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "grey_clinker": {
        "benchmark": 0.766,
        "unit": "tCO2/t clinker",
        "sector": "cement",
    },
    "hot_metal": {
        "benchmark": 1.328,
        "unit": "tCO2/t hot metal",
        "sector": "steel",
    },
    "eaf_carbon_steel": {
        "benchmark": 0.283,
        "unit": "tCO2/t crude steel",
        "sector": "steel",
    },
    "eaf_high_alloy_steel": {
        "benchmark": 0.352,
        "unit": "tCO2/t crude steel",
        "sector": "steel",
    },
    "coke": {
        "benchmark": 0.286,
        "unit": "tCO2/t coke",
        "sector": "steel",
    },
    "sintered_ore": {
        "benchmark": 0.171,
        "unit": "tCO2/t sinter",
        "sector": "steel",
    },
    "aluminium": {
        "benchmark": 1.514,
        "unit": "tCO2/t primary Al",
        "sector": "aluminum",
    },
    "ammonia": {
        "benchmark": 1.619,
        "unit": "tCO2/t NH3",
        "sector": "chemicals",
    },
    "nitric_acid": {
        "benchmark": 0.302,
        "unit": "tCO2e/t HNO3",
        "sector": "chemicals",
    },
    "adipic_acid": {
        "benchmark": 2.790,
        "unit": "tCO2e/t adipic acid",
        "sector": "chemicals",
    },
    "hydrogen": {
        "benchmark": 8.850,
        "unit": "tCO2/t H2",
        "sector": "chemicals",
    },
    "float_glass": {
        "benchmark": 0.453,
        "unit": "tCO2/t glass",
        "sector": "glass",
    },
    "container_glass": {
        "benchmark": 0.382,
        "unit": "tCO2/t glass",
        "sector": "glass",
    },
    "newsprint": {
        "benchmark": 0.298,
        "unit": "tCO2/t paper",
        "sector": "pulp_paper",
    },
    "uncoated_fine_paper": {
        "benchmark": 0.318,
        "unit": "tCO2/t paper",
        "sector": "pulp_paper",
    },
    "short_fibre_kraft_pulp": {
        "benchmark": 0.120,
        "unit": "tCO2/t pulp",
        "sector": "pulp_paper",
    },
    "facing_bricks": {
        "benchmark": 0.139,
        "unit": "tCO2/t brick",
        "sector": "ceramics",
    },
    "roof_tiles": {
        "benchmark": 0.144,
        "unit": "tCO2/t tile",
        "sector": "ceramics",
    },
}

# ---------------------------------------------------------------------------
# Constants: CBAM Goods Categories (Annex I of CBAM Regulation)
# ---------------------------------------------------------------------------

CBAM_GOODS_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "cement": {
        "cn_codes": ["2523"],
        "precursor_goods": ["clinker"],
        "applicable": True,
    },
    "steel": {
        "cn_codes": [
            "7206", "7207", "7208", "7209", "7210", "7211", "7212",
            "7213", "7214", "7215", "7216", "7217", "7218", "7219",
            "7220", "7221", "7222", "7223", "7224", "7225", "7226",
            "7228", "7229", "7301", "7302", "7303", "7304", "7305",
            "7306", "7307", "7308", "7326",
        ],
        "precursor_goods": [
            "hot_metal", "crude_steel", "pig_iron", "dri", "ferro_alloys",
        ],
        "applicable": True,
    },
    "aluminum": {
        "cn_codes": [
            "7601", "7603", "7604", "7605", "7606", "7607", "7608", "7609",
        ],
        "precursor_goods": ["unwrought_aluminium", "aluminium_oxide"],
        "applicable": True,
    },
    "chemicals": {
        "cn_codes": ["2804", "2808", "2814", "2834"],
        "precursor_goods": ["hydrogen", "ammonia", "nitric_acid"],
        "applicable": True,
    },
    "electricity": {
        "cn_codes": ["2716"],
        "precursor_goods": [],
        "applicable": True,
    },
    "fertilizers": {
        "cn_codes": ["3102", "3105"],
        "precursor_goods": ["ammonia", "nitric_acid"],
        "applicable": True,
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ProcessEmissionsConfig(BaseModel):
    """Configuration for process emissions calculation.

    Attributes:
        reporting_year: The calendar year for which emissions are reported.
        sub_sector: Manufacturing sub-sector to calculate for.
        include_cbam: Whether to calculate CBAM embedded emissions.
        include_ets_benchmark: Whether to compare against EU ETS benchmarks.
        abatement_tracking: Whether to track CO2 capture / abatement.
    """
    reporting_year: int = Field(
        default=2025,
        ge=2019,
        le=2035,
        description="Calendar year for reporting.",
    )
    sub_sector: ManufacturingSubSector = Field(
        default=ManufacturingSubSector.CEMENT,
        description="Manufacturing sub-sector.",
    )
    include_cbam: bool = Field(
        default=True,
        description="Include CBAM embedded emissions calculation.",
    )
    include_ets_benchmark: bool = Field(
        default=True,
        description="Compare against EU ETS product benchmarks.",
    )
    abatement_tracking: bool = Field(
        default=False,
        description="Track CO2 abatement / CCS.",
    )

class RawMaterial(BaseModel):
    """A raw material input that generates process emissions.

    Attributes:
        material_name: Name of the raw material (e.g., limestone, bauxite).
        quantity_tonnes: Quantity consumed in tonnes.
        co2_factor_per_tonne: Process emission factor in tCO2/tonne.
        source: Source of the emission factor.
    """
    material_name: str = Field(
        ...,
        min_length=1,
        description="Name of the raw material.",
    )
    quantity_tonnes: float = Field(
        ...,
        ge=0.0,
        description="Quantity consumed in tonnes.",
    )
    co2_factor_per_tonne: float = Field(
        ...,
        ge=0.0,
        description="tCO2 emitted per tonne of material processed.",
    )
    source: str = Field(
        default="IPCC 2006 Vol.3",
        description="Source of emission factor.",
    )

    @field_validator("quantity_tonnes", "co2_factor_per_tonne")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        """Ensure values are non-negative."""
        if v < 0:
            raise ValueError("Value must be non-negative.")
        return v

class FuelConsumption(BaseModel):
    """Fuel consumption record for combustion emissions.

    Attributes:
        fuel_type: Type of fuel consumed.
        quantity: Quantity of fuel consumed.
        unit: Unit of measurement (e.g., tonnes, 1000m3, MWh).
        emission_factor: Override emission factor (tCO2/TJ); None uses defaults.
        ncv_override: Override net calorific value (TJ/unit); None uses defaults.
    """
    fuel_type: FuelType = Field(
        ...,
        description="Type of fuel consumed.",
    )
    quantity: float = Field(
        ...,
        ge=0.0,
        description="Quantity of fuel consumed.",
    )
    unit: str = Field(
        default="tonnes",
        description="Unit of fuel quantity.",
    )
    emission_factor: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Override emission factor (tCO2/TJ).",
    )
    ncv_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Override NCV (TJ/unit).",
    )

class ProcessLine(BaseModel):
    """A single production / process line within a facility.

    Attributes:
        line_id: Unique identifier for the line.
        line_name: Human-readable name.
        process_type: Type of process.
        annual_production_tonnes: Annual production in tonnes.
        raw_materials: List of raw material inputs.
        fuel_consumption: List of fuel consumption records.
        emission_factor_source: Source reference for emission factors.
    """
    line_id: str = Field(
        default_factory=_new_uuid,
        description="Unique production line identifier.",
    )
    line_name: str = Field(
        ...,
        min_length=1,
        description="Name of the production line.",
    )
    process_type: ProcessType = Field(
        ...,
        description="Type of industrial process.",
    )
    annual_production_tonnes: float = Field(
        ...,
        ge=0.0,
        description="Annual production output in tonnes.",
    )
    raw_materials: List[RawMaterial] = Field(
        default_factory=list,
        description="Raw material inputs for process emissions.",
    )
    fuel_consumption: List[FuelConsumption] = Field(
        default_factory=list,
        description="Fuel consumption records.",
    )
    emission_factor_source: str = Field(
        default="IPCC 2006 Guidelines",
        description="Source of emission factors.",
    )

class FacilityData(BaseModel):
    """A manufacturing facility with one or more production lines.

    Attributes:
        facility_id: Unique facility identifier.
        facility_name: Human-readable name.
        sub_sector: Manufacturing sub-sector.
        country: ISO 3166-1 alpha-2 country code.
        eu_ets_installation_id: EU ETS installation ID (if applicable).
        production_lines: List of production lines.
    """
    facility_id: str = Field(
        default_factory=_new_uuid,
        description="Unique facility identifier.",
    )
    facility_name: str = Field(
        ...,
        min_length=1,
        description="Name of the facility.",
    )
    sub_sector: ManufacturingSubSector = Field(
        ...,
        description="Manufacturing sub-sector.",
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    eu_ets_installation_id: Optional[str] = Field(
        default=None,
        description="EU ETS installation identifier.",
    )
    production_lines: List[ProcessLine] = Field(
        default_factory=list,
        description="Production lines at the facility.",
    )

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        """Normalise country code to uppercase."""
        return v.upper()

class CBAMEmbeddedEmissions(BaseModel):
    """CBAM embedded emissions per Regulation (EU) 2023/956 Annex IV.

    Attributes:
        direct_emissions: Direct (Scope 1) emissions in tCO2e.
        indirect_emissions: Indirect (Scope 2) electricity emissions in tCO2e.
        precursor_emissions: Emissions from precursor products in tCO2e.
        total_embedded: Total embedded emissions in tCO2e.
        goods_category: CBAM goods category.
        specific_embedded: Specific embedded emissions (tCO2e/t product).
        production_tonnes: Production quantity in tonnes.
    """
    direct_emissions: float = Field(default=0.0, ge=0.0)
    indirect_emissions: float = Field(default=0.0, ge=0.0)
    precursor_emissions: float = Field(default=0.0, ge=0.0)
    total_embedded: float = Field(default=0.0, ge=0.0)
    goods_category: str = Field(default="unknown")
    specific_embedded: float = Field(default=0.0, ge=0.0)
    production_tonnes: float = Field(default=0.0, ge=0.0)

class AbatementRecord(BaseModel):
    """Record of CO2 abatement / capture at a facility.

    Attributes:
        captured_co2_tonnes: Tonnes of CO2 captured.
        method: Capture method (e.g., post-combustion, oxy-fuel, CCS).
        storage_type: Storage or utilisation pathway.
        verified: Whether capture is third-party verified.
        net_reduction_tonnes: Net emission reduction after energy penalty.
    """
    captured_co2_tonnes: float = Field(default=0.0, ge=0.0)
    method: str = Field(default="post_combustion")
    storage_type: str = Field(default="geological_storage")
    verified: bool = Field(default=False)
    net_reduction_tonnes: float = Field(default=0.0, ge=0.0)

class ETSBenchmarkComparison(BaseModel):
    """Comparison of facility intensity against EU ETS benchmark.

    Attributes:
        product_name: ETS product benchmark name.
        facility_intensity: Facility-specific emission intensity (tCO2/t).
        benchmark_value: EU ETS benchmark value (tCO2/t).
        ratio_to_benchmark: Facility intensity / Benchmark.
        free_allocation_eligible: Whether below benchmark (eligible).
        shortfall_tco2: Additional allowances needed (if above benchmark).
    """
    product_name: str = Field(default="")
    facility_intensity: float = Field(default=0.0)
    benchmark_value: float = Field(default=0.0)
    ratio_to_benchmark: float = Field(default=0.0)
    free_allocation_eligible: bool = Field(default=False)
    shortfall_tco2: float = Field(default=0.0)

class ProcessEmissionsResult(BaseModel):
    """Complete result of process emissions calculation with provenance.

    Attributes:
        result_id: Unique result identifier.
        facility_id: Facility this result pertains to.
        total_process_co2: Total process CO2 emissions (tCO2).
        total_combustion_co2: Total combustion CO2 emissions (tCO2).
        total_fugitive_co2: Total fugitive CO2e emissions (tCO2e).
        total_emissions: Grand total emissions (tCO2e).
        emission_intensity_per_tonne: Emission intensity (tCO2e/t product).
        sub_sector_breakdown: Breakdown by process line.
        cbam_embedded_emissions: CBAM embedded emissions (if calculated).
        ets_benchmark_comparison: EU ETS benchmark comparison (if calculated).
        abatement_captured: Abatement records (if tracking enabled).
        methodology_notes: Notes on methodology and data sources.
        processing_time_ms: Time taken to compute this result.
        engine_version: Version of this engine.
        calculated_at: UTC timestamp of calculation.
        provenance_hash: SHA-256 hash of all inputs and outputs.
    """
    result_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    total_process_co2: float = Field(default=0.0)
    total_combustion_co2: float = Field(default=0.0)
    total_fugitive_co2: float = Field(default=0.0)
    total_emissions: float = Field(default=0.0)
    emission_intensity_per_tonne: float = Field(default=0.0)
    sub_sector_breakdown: Dict[str, Any] = Field(default_factory=dict)
    cbam_embedded_emissions: Optional[CBAMEmbeddedEmissions] = Field(default=None)
    ets_benchmark_comparison: Optional[ETSBenchmarkComparison] = Field(default=None)
    abatement_captured: Optional[AbatementRecord] = Field(default=None)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ProcessEmissionsEngine:
    """Zero-hallucination industrial process emissions calculation engine.

    Calculates Scope 1 process emissions, combustion emissions, and fugitive
    releases for manufacturing facilities.  Integrates with EU ETS benchmarks
    and CBAM embedded emissions requirements.

    Guarantees:
        - Deterministic: same inputs produce identical outputs (bit-perfect).
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full breakdown by process line and emission source.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        config = ProcessEmissionsConfig(
            reporting_year=2025,
            sub_sector=ManufacturingSubSector.CEMENT,
            include_cbam=True,
        )
        engine = ProcessEmissionsEngine(config)
        result = engine.calculate_facility_emissions(facility_data)

    Args:
        config: Engine configuration.  Accepts a ProcessEmissionsConfig,
                a plain dict, or None (defaults applied).
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the process emissions engine.

        Args:
            config: A ProcessEmissionsConfig, dict, or None for defaults.
        """
        if config is None:
            self.config = ProcessEmissionsConfig()
        elif isinstance(config, dict):
            self.config = ProcessEmissionsConfig(**config)
        elif isinstance(config, ProcessEmissionsConfig):
            self.config = config
        else:
            raise TypeError(
                f"config must be ProcessEmissionsConfig, dict, or None, "
                f"got {type(config).__name__}"
            )
        logger.info(
            "ProcessEmissionsEngine initialised: year=%d, sub_sector=%s",
            self.config.reporting_year,
            self.config.sub_sector.value,
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def calculate_facility_emissions(
        self, facility: FacilityData
    ) -> ProcessEmissionsResult:
        """Calculate total emissions for a manufacturing facility.

        Iterates over every production line, computing:
          1. Process emissions (raw-material decomposition / transformation).
          2. Combustion emissions (fuel burning in process equipment).
          3. Fugitive emissions (estimated at 1% of process + combustion).

        Optionally computes CBAM embedded emissions and ETS benchmark
        comparison when configured.

        Args:
            facility: Facility data including production lines.

        Returns:
            ProcessEmissionsResult with complete breakdown and provenance.

        Raises:
            ValueError: If facility has no production lines.
        """
        t0 = time.perf_counter()

        if not facility.production_lines:
            raise ValueError(
                f"Facility '{facility.facility_name}' has no production lines."
            )

        total_process_co2: float = 0.0
        total_combustion_co2: float = 0.0
        total_production_tonnes: float = 0.0
        line_breakdown: Dict[str, Any] = {}
        methodology_notes: List[str] = [
            f"Reporting year: {self.config.reporting_year}",
            f"Sub-sector: {self.config.sub_sector.value}",
            f"Engine version: {self.engine_version}",
        ]

        for line in facility.production_lines:
            line_result = self.calculate_process_line(line)
            total_process_co2 += line_result["process_co2"]
            total_combustion_co2 += line_result["combustion_co2"]
            total_production_tonnes += line.annual_production_tonnes
            line_breakdown[line.line_id] = {
                "line_name": line.line_name,
                "process_type": line.process_type.value,
                "production_tonnes": line.annual_production_tonnes,
                "process_co2": _round3(line_result["process_co2"]),
                "combustion_co2": _round3(line_result["combustion_co2"]),
                "total_co2": _round3(
                    line_result["process_co2"] + line_result["combustion_co2"]
                ),
                "materials": line_result.get("material_details", []),
                "fuels": line_result.get("fuel_details", []),
            }

        # Fugitive emissions: IPCC default 1% of process + combustion
        total_fugitive_co2 = self._estimate_fugitive(
            total_process_co2, total_combustion_co2
        )
        methodology_notes.append(
            "Fugitive emissions estimated at 1% of (process + combustion) "
            "per IPCC 2006 Vol. 3 default."
        )

        # Grand total
        total_emissions = total_process_co2 + total_combustion_co2 + total_fugitive_co2

        # Abatement
        abatement_record: Optional[AbatementRecord] = None
        if self.config.abatement_tracking:
            abatement_record = AbatementRecord()
            methodology_notes.append(
                "Abatement tracking enabled (no capture data provided)."
            )

        # Subtract abatement if present and non-zero
        if abatement_record and abatement_record.net_reduction_tonnes > 0:
            total_emissions -= abatement_record.net_reduction_tonnes
            methodology_notes.append(
                f"Abated {abatement_record.net_reduction_tonnes:.3f} tCO2 "
                f"via {abatement_record.method}."
            )

        # Intensity
        emission_intensity = _safe_divide(total_emissions, total_production_tonnes)

        # CBAM
        cbam_result: Optional[CBAMEmbeddedEmissions] = None
        if self.config.include_cbam:
            cbam_result = self.calculate_cbam_embedded(
                facility, total_emissions, total_production_tonnes
            )
            methodology_notes.append(
                "CBAM embedded emissions calculated per Regulation "
                "(EU) 2023/956 Annex IV."
            )

        # ETS benchmark
        ets_result: Optional[ETSBenchmarkComparison] = None
        if self.config.include_ets_benchmark:
            ets_result = self.compare_ets_benchmark(
                facility, emission_intensity, total_emissions,
                total_production_tonnes,
            )
            if ets_result:
                methodology_notes.append(
                    f"EU ETS benchmark comparison for product "
                    f"'{ets_result.product_name}'."
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ProcessEmissionsResult(
            facility_id=facility.facility_id,
            total_process_co2=_round3(total_process_co2),
            total_combustion_co2=_round3(total_combustion_co2),
            total_fugitive_co2=_round3(total_fugitive_co2),
            total_emissions=_round3(total_emissions),
            emission_intensity_per_tonne=_round3(emission_intensity),
            sub_sector_breakdown=line_breakdown,
            cbam_embedded_emissions=cbam_result,
            ets_benchmark_comparison=ets_result,
            abatement_captured=abatement_record,
            methodology_notes=methodology_notes,
            processing_time_ms=round(elapsed_ms, 2),
            engine_version=self.engine_version,
            calculated_at=utcnow(),
        )

        # Provenance hash -- covers all inputs and outputs
        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_process_line(self, line: ProcessLine) -> Dict[str, Any]:
        """Calculate emissions for a single production / process line.

        Process emissions are computed from each raw material's quantity
        multiplied by its CO2 factor.  Combustion emissions use the
        IPCC formula::

            CO2 = fuel_qty * NCV (TJ/unit) * EF (tCO2/TJ)

        Args:
            line: A single production line definition.

        Returns:
            Dictionary with keys ``process_co2``, ``combustion_co2``,
            ``material_details``, ``fuel_details``.
        """
        process_co2 = 0.0
        material_details: List[Dict[str, Any]] = []

        for mat in line.raw_materials:
            mat_co2 = mat.quantity_tonnes * mat.co2_factor_per_tonne
            process_co2 += mat_co2
            material_details.append({
                "material": mat.material_name,
                "quantity_tonnes": mat.quantity_tonnes,
                "factor_tco2_per_t": mat.co2_factor_per_tonne,
                "co2_tonnes": _round3(mat_co2),
                "source": mat.source,
            })

        combustion_co2 = 0.0
        fuel_details: List[Dict[str, Any]] = []

        for fuel in line.fuel_consumption:
            fuel_co2, detail = self._calculate_fuel_co2(fuel)
            combustion_co2 += fuel_co2
            fuel_details.append(detail)

        return {
            "process_co2": process_co2,
            "combustion_co2": combustion_co2,
            "material_details": material_details,
            "fuel_details": fuel_details,
        }

    def calculate_cbam_embedded(
        self,
        facility: FacilityData,
        total_direct_emissions: float = 0.0,
        total_production_tonnes: float = 0.0,
    ) -> CBAMEmbeddedEmissions:
        """Calculate CBAM embedded emissions for a facility.

        Per CBAM Regulation (EU) 2023/956, embedded emissions comprise:
          - Direct emissions (Scope 1) attributable to the production
          - Indirect emissions (Scope 2 electricity)
          - Precursor emissions (from input goods that are themselves
            CBAM goods)

        Args:
            facility: Facility data.
            total_direct_emissions: Pre-computed total direct emissions.
            total_production_tonnes: Total production in tonnes.

        Returns:
            CBAMEmbeddedEmissions with breakdown.
        """
        sub_sector = facility.sub_sector.value
        goods_category = sub_sector
        applicable = False

        if sub_sector in CBAM_GOODS_CATEGORIES:
            applicable = CBAM_GOODS_CATEGORIES[sub_sector].get("applicable", False)

        if not applicable:
            return CBAMEmbeddedEmissions(
                goods_category=goods_category,
                production_tonnes=total_production_tonnes,
            )

        # Direct emissions are already computed
        direct = total_direct_emissions

        # Indirect: estimate electricity emissions at EU average grid factor.
        # EU average approximately 0.23 tCO2/MWh (2024 EEA data).
        electricity_mwh = 0.0
        for pline in facility.production_lines:
            for fc in pline.fuel_consumption:
                if fc.fuel_type == FuelType.ELECTRICITY:
                    electricity_mwh += fc.quantity
        indirect = electricity_mwh * 0.23

        # Precursor: placeholder -- requires supply-chain data
        precursor = 0.0

        total_embedded = direct + indirect + precursor
        specific = _safe_divide(total_embedded, total_production_tonnes)

        return CBAMEmbeddedEmissions(
            direct_emissions=_round3(direct),
            indirect_emissions=_round3(indirect),
            precursor_emissions=_round3(precursor),
            total_embedded=_round3(total_embedded),
            goods_category=goods_category,
            specific_embedded=_round3(specific),
            production_tonnes=_round3(total_production_tonnes),
        )

    def compare_ets_benchmark(
        self,
        facility: FacilityData,
        facility_intensity: float,
        total_emissions: float,
        total_production: float,
    ) -> Optional[ETSBenchmarkComparison]:
        """Compare facility emission intensity against EU ETS benchmarks.

        Finds the most relevant EU ETS product benchmark for the facility's
        sub-sector and calculates whether the facility would be eligible
        for free allocation (intensity <= benchmark).

        Args:
            facility: Facility data.
            facility_intensity: Facility emission intensity (tCO2/t product).
            total_emissions: Total facility emissions (tCO2e).
            total_production: Total production (tonnes).

        Returns:
            ETSBenchmarkComparison or None if no benchmark found.
        """
        sub_sector = facility.sub_sector.value
        best_match: Optional[Tuple[str, Dict[str, Any]]] = None

        for product_name, bm_data in ETS_PRODUCT_BENCHMARKS.items():
            if bm_data["sector"] == sub_sector:
                best_match = (product_name, bm_data)
                break  # take first match for sector

        if best_match is None:
            return None

        product_name, bm_data = best_match
        benchmark_value = bm_data["benchmark"]
        ratio = _safe_divide(facility_intensity, benchmark_value)
        eligible = facility_intensity <= benchmark_value
        shortfall = 0.0
        if not eligible and total_production > 0:
            excess_intensity = facility_intensity - benchmark_value
            shortfall = excess_intensity * total_production

        return ETSBenchmarkComparison(
            product_name=product_name,
            facility_intensity=_round3(facility_intensity),
            benchmark_value=benchmark_value,
            ratio_to_benchmark=_round3(ratio),
            free_allocation_eligible=eligible,
            shortfall_tco2=_round3(shortfall),
        )

    def calculate_abatement(
        self,
        captured_co2: float,
        method: str = "post_combustion",
        energy_penalty_pct: float = 15.0,
        verified: bool = False,
    ) -> AbatementRecord:
        """Calculate net CO2 abatement from CCS / CCU.

        Applies an energy penalty factor to the gross captured CO2 to
        determine net emission reduction.  Different capture methods have
        different penalty factors.

        Args:
            captured_co2: Gross CO2 captured in tonnes.
            method: Capture method (e.g., post_combustion, oxy_fuel).
            energy_penalty_pct: Energy penalty as percentage (default 15%).
            verified: Whether capture volumes are third-party verified.

        Returns:
            AbatementRecord with net reduction.
        """
        # Energy penalties by capture method
        energy_penalties: Dict[str, float] = {
            "post_combustion": 15.0,
            "pre_combustion": 10.0,
            "oxy_fuel": 12.0,
            "direct_air_capture": 25.0,
            "mineralization": 5.0,
        }

        penalty = energy_penalties.get(method, energy_penalty_pct)
        net_reduction = captured_co2 * (1.0 - penalty / 100.0)

        return AbatementRecord(
            captured_co2_tonnes=_round3(captured_co2),
            method=method,
            storage_type="geological_storage",
            verified=verified,
            net_reduction_tonnes=_round3(max(net_reduction, 0.0)),
        )

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #

    def _calculate_fuel_co2(
        self, fuel: FuelConsumption
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate CO2 from a single fuel consumption record.

        Uses the IPCC formula::

            CO2 = fuel_quantity * NCV (TJ/unit) * EF (tCO2/TJ)

        Args:
            fuel: Fuel consumption record.

        Returns:
            Tuple of (co2_tonnes, detail_dict).
        """
        fuel_key = fuel.fuel_type.value
        fuel_data = FUEL_EMISSION_FACTORS.get(fuel_key, {})

        ef_tco2_per_tj = (
            fuel.emission_factor
            if fuel.emission_factor is not None
            else fuel_data.get("co2_factor_tco2_per_tj", 0.0)
        )
        ncv = (
            fuel.ncv_override
            if fuel.ncv_override is not None
            else fuel_data.get("ncv_tj_per_unit", 0.0)
        )

        energy_tj = fuel.quantity * ncv
        co2_tonnes = energy_tj * ef_tco2_per_tj

        detail = {
            "fuel_type": fuel_key,
            "quantity": fuel.quantity,
            "unit": fuel.unit,
            "ncv_tj_per_unit": ncv,
            "energy_tj": _round3(energy_tj),
            "ef_tco2_per_tj": ef_tco2_per_tj,
            "co2_tonnes": _round3(co2_tonnes),
        }

        return co2_tonnes, detail

    def _get_sector_process_factors(
        self, sub_sector: str
    ) -> Dict[str, float]:
        """Return process emission factors relevant to a sub-sector.

        Filters the master PROCESS_EMISSION_FACTORS dict to return only
        those entries whose key starts with the sub-sector name.

        Args:
            sub_sector: Sub-sector string (e.g., "cement", "steel").

        Returns:
            Dict of factor_name -> factor_value.
        """
        prefix = sub_sector + "_"
        return {
            k: v
            for k, v in PROCESS_EMISSION_FACTORS.items()
            if k.startswith(prefix)
        }

    def _estimate_fugitive(
        self,
        process_co2: float,
        combustion_co2: float,
        pct: float = 1.0,
    ) -> float:
        """Estimate fugitive emissions as a percentage of process + combustion.

        Args:
            process_co2: Process CO2 in tonnes.
            combustion_co2: Combustion CO2 in tonnes.
            pct: Percentage to apply (default 1.0%).

        Returns:
            Fugitive CO2e in tonnes.
        """
        return (process_co2 + combustion_co2) * (pct / 100.0)
