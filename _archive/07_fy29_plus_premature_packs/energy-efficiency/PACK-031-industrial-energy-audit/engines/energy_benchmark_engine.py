# -*- coding: utf-8 -*-
"""
EnergyBenchmarkEngine - PACK-031 Industrial Energy Audit Engine 10
====================================================================

Benchmarks facility energy performance against industry peers, EU BAT-AEL
(Best Available Techniques - Associated Energy Levels) from BREF documents,
and assigns A-G energy performance ratings.  Calculates Specific Energy
Consumption (SEC), percentile ranking within sector peer groups, gap-to-
best-practice with quantified savings potential, improvement trajectories,
and carbon intensity benchmarking alongside energy intensity.

Scope:
    - Specific Energy Consumption (SEC) per production unit
    - EnPI benchmarking against sector averages and best practice
    - EU BAT-AEL comparison from BREF documents
    - Percentile ranking within sector peer group
    - Gap-to-best-practice analysis with quantified savings potential
    - Multi-dimensional benchmarking (sector, size, geography, product mix)
    - Energy performance scoring (A-G rating like EU energy labels)
    - Improvement trajectory tracking (year-over-year)
    - Carbon intensity benchmarking (kgCO2/unit alongside kWh/unit)

Regulatory / Standard References:
    - ISO 50001:2018 Energy Management Systems (EnPIs)
    - ISO 50006:2014 Measuring energy performance using baselines
    - EU BAT Reference Documents (BREF) per IED 2010/75/EU
    - EN 16247-1:2022 Energy Audits (general requirements)
    - EU Energy Efficiency Directive 2023/1791
    - ESRS E1-5 (Energy consumption and mix)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - SEC benchmarks from published EU BREF documents and IEA data
    - BAT-AEL ranges from official BAT Conclusions
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-031 Industrial Energy Audit
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(num: Decimal, den: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    return default if den == Decimal("0") else num / den

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* decimal digits and return float."""
    return float(value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))

def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round1(value: Any) -> float:
    """Round to 1 decimal place."""
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IndustrySector(str, Enum):
    """Industrial sector classifications for benchmarking."""
    CEMENT = "cement"
    STEEL_INTEGRATED = "steel_integrated"
    STEEL_EAF = "steel_eaf"
    ALUMINIUM_PRIMARY = "aluminium_primary"
    ALUMINIUM_SECONDARY = "aluminium_secondary"
    GLASS_FLAT = "glass_flat"
    GLASS_CONTAINER = "glass_container"
    CERAMICS = "ceramics"
    PULP_PAPER_INTEGRATED = "pulp_paper_integrated"
    PULP_PAPER_RECYCLED = "pulp_paper_recycled"
    FOOD_DAIRY = "food_dairy"
    FOOD_MEAT = "food_meat"
    FOOD_BEVERAGE = "food_beverage"
    FOOD_BAKERY = "food_bakery"
    FOOD_FROZEN = "food_frozen"
    TEXTILES_FINISHING = "textiles_finishing"
    TEXTILES_SPINNING = "textiles_spinning"
    CHEMICALS_BULK = "chemicals_bulk"
    CHEMICALS_SPECIALTY = "chemicals_specialty"
    PHARMACEUTICALS = "pharmaceuticals"
    PLASTICS_INJECTION = "plastics_injection"
    PLASTICS_EXTRUSION = "plastics_extrusion"
    AUTOMOTIVE_ASSEMBLY = "automotive_assembly"
    AUTOMOTIVE_COMPONENTS = "automotive_components"
    ELECTRONICS = "electronics"
    MACHINERY = "machinery"
    RUBBER = "rubber"
    WOOD_PRODUCTS = "wood_products"
    PRINTING = "printing"
    DATA_CENTRE = "data_centre"
    COLD_STORAGE = "cold_storage"
    GENERIC_MANUFACTURING = "generic_manufacturing"

class EnergyRatingClass(str, Enum):
    """EU-style energy performance rating classes (A-G)."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

class BREFDocument(str, Enum):
    """EU BREF reference documents for BAT-AEL benchmarking."""
    CLM = "clm"        # Cement, Lime and Magnesium Oxide
    IS = "is"          # Iron and Steel
    NFM = "nfm"        # Non-Ferrous Metals
    GLS = "gls"        # Glass Manufacturing
    CER = "cer"        # Ceramics
    PP = "pp"          # Pulp and Paper
    FDM = "fdm"        # Food, Drink and Milk
    TXT = "txt"        # Textiles
    LVOC = "lvoc"      # Large Volume Organic Chemicals
    SIC = "sic"        # Speciality Inorganic Chemicals
    OFC = "ofc"        # Organic Fine Chemicals
    POL = "pol"        # Polymers
    STM = "stm"       # Surface Treatment of Metals
    SA = "sa"          # Smitheries and Foundries
    ENE = "ene"        # Energy Efficiency

class BenchmarkMetric(str, Enum):
    """Benchmark comparison metrics."""
    SEC_KWH_PER_TONNE = "sec_kwh_per_tonne"
    SEC_KWH_PER_UNIT = "sec_kwh_per_unit"
    SEC_MJ_PER_TONNE = "sec_mj_per_tonne"
    SEC_KWH_PER_SQM = "sec_kwh_per_sqm"
    CARBON_INTENSITY_KG_PER_TONNE = "carbon_intensity_kg_co2_per_tonne"
    CARBON_INTENSITY_KG_PER_UNIT = "carbon_intensity_kg_co2_per_unit"
    EUI_KWH_PER_SQM = "eui_kwh_per_sqm"
    COST_EUR_PER_TONNE = "cost_eur_per_tonne"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Specific Energy Consumption (SEC) benchmarks by industrial sector.
# Values in kWh/tonne of product (except where noted).
# Source: EU BREF documents, IEA Industrial Energy Efficiency reports,
# US DOE Manufacturing Energy Consumption Survey (MECS), national
# energy agency sector benchmarking reports.
SEC_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    IndustrySector.CEMENT: {
        "best_practice": 780,    # kWh/t clinker (dry process)
        "good": 860,
        "average": 970,
        "poor": 1200,
        "unit": "kWh/tonne_clinker",
        "bref": BREFDocument.CLM,
        "source": "CLM BREF 2013, IEA Cement Technology Roadmap",
    },
    IndustrySector.STEEL_INTEGRATED: {
        "best_practice": 4700,
        "good": 5200,
        "average": 5800,
        "poor": 7000,
        "unit": "kWh/tonne_crude_steel",
        "bref": BREFDocument.IS,
        "source": "IS BREF 2012, worldsteel Energy Use in the Steel Industry",
    },
    IndustrySector.STEEL_EAF: {
        "best_practice": 400,
        "good": 500,
        "average": 620,
        "poor": 800,
        "unit": "kWh/tonne_crude_steel",
        "bref": BREFDocument.IS,
        "source": "IS BREF 2012 BAT 47",
    },
    IndustrySector.ALUMINIUM_PRIMARY: {
        "best_practice": 13000,
        "good": 14000,
        "average": 15200,
        "poor": 17000,
        "unit": "kWh/tonne_Al",
        "bref": BREFDocument.NFM,
        "source": "NFM BREF 2014 BAT 85, IAI sector data",
    },
    IndustrySector.ALUMINIUM_SECONDARY: {
        "best_practice": 500,
        "good": 700,
        "average": 900,
        "poor": 1200,
        "unit": "kWh/tonne_Al",
        "bref": BREFDocument.NFM,
        "source": "NFM BREF 2014, recycled aluminium",
    },
    IndustrySector.GLASS_FLAT: {
        "best_practice": 1530,
        "good": 1700,
        "average": 2080,
        "poor": 2500,
        "unit": "kWh/tonne_glass",
        "bref": BREFDocument.GLS,
        "source": "GLS BREF 2012 BAT 4",
    },
    IndustrySector.GLASS_CONTAINER: {
        "best_practice": 1110,
        "good": 1300,
        "average": 1670,
        "poor": 2000,
        "unit": "kWh/tonne_glass",
        "bref": BREFDocument.GLS,
        "source": "GLS BREF 2012 BAT 4",
    },
    IndustrySector.CERAMICS: {
        "best_practice": 556,
        "good": 700,
        "average": 970,
        "poor": 1300,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.CER,
        "source": "CER BREF 2007 BAT 12",
    },
    IndustrySector.PULP_PAPER_INTEGRATED: {
        "best_practice": 2780,
        "good": 3200,
        "average": 3890,
        "poor": 4800,
        "unit": "kWh/tonne_paper",
        "bref": BREFDocument.PP,
        "source": "PP BREF 2015 BAT 8, CEPI sector data",
    },
    IndustrySector.PULP_PAPER_RECYCLED: {
        "best_practice": 1000,
        "good": 1300,
        "average": 1700,
        "poor": 2200,
        "unit": "kWh/tonne_paper",
        "bref": BREFDocument.PP,
        "source": "PP BREF 2015, recycled fibre",
    },
    IndustrySector.FOOD_DAIRY: {
        "best_practice": 180,
        "good": 250,
        "average": 380,
        "poor": 550,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.FDM,
        "source": "FDM BREF 2006 BAT 5, Carbon Trust sector guide",
    },
    IndustrySector.FOOD_MEAT: {
        "best_practice": 300,
        "good": 420,
        "average": 600,
        "poor": 900,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.FDM,
        "source": "FDM BREF 2006, meat processing",
    },
    IndustrySector.FOOD_BEVERAGE: {
        "best_practice": 80,
        "good": 120,
        "average": 200,
        "poor": 350,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.FDM,
        "source": "FDM BREF 2006, beverage sector",
    },
    IndustrySector.FOOD_BAKERY: {
        "best_practice": 350,
        "good": 500,
        "average": 750,
        "poor": 1100,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.FDM,
        "source": "FDM BREF 2006, bakery sector",
    },
    IndustrySector.FOOD_FROZEN: {
        "best_practice": 400,
        "good": 550,
        "average": 800,
        "poor": 1200,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.FDM,
        "source": "FDM BREF 2006, frozen food sector",
    },
    IndustrySector.TEXTILES_FINISHING: {
        "best_practice": 2220,
        "good": 2800,
        "average": 4170,
        "poor": 6000,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.TXT,
        "source": "TXT BREF 2003 BAT 10",
    },
    IndustrySector.TEXTILES_SPINNING: {
        "best_practice": 800,
        "good": 1100,
        "average": 1500,
        "poor": 2200,
        "unit": "kWh/tonne_yarn",
        "bref": BREFDocument.TXT,
        "source": "TXT BREF 2003, spinning operations",
    },
    IndustrySector.CHEMICALS_BULK: {
        "best_practice": 1390,
        "good": 1700,
        "average": 2220,
        "poor": 3000,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.LVOC,
        "source": "LVOC BREF 2017 BAT 5",
    },
    IndustrySector.CHEMICALS_SPECIALTY: {
        "best_practice": 2500,
        "good": 3500,
        "average": 5000,
        "poor": 8000,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.SIC,
        "source": "SIC BREF 2007, specialty chemicals",
    },
    IndustrySector.PHARMACEUTICALS: {
        "best_practice": 5560,
        "good": 7000,
        "average": 13890,
        "poor": 20000,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.OFC,
        "source": "OFC BREF 2006 BAT 3, pharma sector benchmarking",
    },
    IndustrySector.PLASTICS_INJECTION: {
        "best_practice": 600,
        "good": 800,
        "average": 1100,
        "poor": 1600,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.POL,
        "source": "POL BREF 2007, injection moulding",
    },
    IndustrySector.PLASTICS_EXTRUSION: {
        "best_practice": 350,
        "good": 500,
        "average": 700,
        "poor": 1000,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.POL,
        "source": "POL BREF 2007, extrusion",
    },
    IndustrySector.AUTOMOTIVE_ASSEMBLY: {
        "best_practice": 700,
        "good": 900,
        "average": 1200,
        "poor": 1800,
        "unit": "kWh/vehicle",
        "bref": BREFDocument.STM,
        "source": "STM BREF 2006 BAT 6, automotive OEM data",
    },
    IndustrySector.AUTOMOTIVE_COMPONENTS: {
        "best_practice": 500,
        "good": 700,
        "average": 1000,
        "poor": 1500,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.STM,
        "source": "STM BREF 2006, automotive components",
    },
    IndustrySector.ELECTRONICS: {
        "best_practice": 6940,
        "good": 8500,
        "average": 11110,
        "poor": 16000,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.ENE,
        "source": "Industry estimate, electronics manufacturing",
    },
    IndustrySector.MACHINERY: {
        "best_practice": 400,
        "good": 600,
        "average": 900,
        "poor": 1400,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.SA,
        "source": "SA BREF 2005, general machinery",
    },
    IndustrySector.RUBBER: {
        "best_practice": 800,
        "good": 1100,
        "average": 1500,
        "poor": 2200,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.POL,
        "source": "POL BREF 2007, rubber products",
    },
    IndustrySector.WOOD_PRODUCTS: {
        "best_practice": 200,
        "good": 300,
        "average": 500,
        "poor": 800,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.ENE,
        "source": "Industry estimate, wood products manufacturing",
    },
    IndustrySector.PRINTING: {
        "best_practice": 300,
        "good": 450,
        "average": 700,
        "poor": 1100,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.ENE,
        "source": "Industry estimate, printing sector",
    },
    IndustrySector.DATA_CENTRE: {
        "best_practice": 1.10,  # PUE
        "good": 1.30,
        "average": 1.58,
        "poor": 2.00,
        "unit": "PUE",
        "bref": BREFDocument.ENE,
        "source": "EU Code of Conduct for Data Centres 2024",
    },
    IndustrySector.COLD_STORAGE: {
        "best_practice": 80,
        "good": 120,
        "average": 180,
        "poor": 280,
        "unit": "kWh/m3/year",
        "bref": BREFDocument.ENE,
        "source": "Global Cold Chain Alliance, IIR benchmarks",
    },
    IndustrySector.GENERIC_MANUFACTURING: {
        "best_practice": 500,
        "good": 800,
        "average": 1200,
        "poor": 2000,
        "unit": "kWh/tonne_product",
        "bref": BREFDocument.ENE,
        "source": "EU ENE BREF, generic manufacturing",
    },
}

# BAT-AEL energy levels by sector (from BREF documents).
# Values in kWh/tonne for direct comparison with SEC.
BAT_AEL_DATABASE: Dict[str, Dict[str, Any]] = {
    IndustrySector.CEMENT: {
        "bat_ael_min_kwh": 720, "bat_ael_max_kwh": 830,
        "bref": "CLM BREF 2013", "bat_reference": "BAT 16",
        "notes": "Dry process with preheater and precalciner, per tonne clinker",
    },
    IndustrySector.STEEL_INTEGRATED: {
        "bat_ael_min_kwh": 4500, "bat_ael_max_kwh": 5000,
        "bref": "IS BREF 2012", "bat_reference": "BAT 1",
        "notes": "BF-BOF route, per tonne crude steel",
    },
    IndustrySector.STEEL_EAF: {
        "bat_ael_min_kwh": 350, "bat_ael_max_kwh": 450,
        "bref": "IS BREF 2012", "bat_reference": "BAT 47",
        "notes": "EAF route, electricity only, per tonne crude steel",
    },
    IndustrySector.ALUMINIUM_PRIMARY: {
        "bat_ael_min_kwh": 12800, "bat_ael_max_kwh": 13500,
        "bref": "NFM BREF 2014", "bat_reference": "BAT 85",
        "notes": "Hall-Heroult process, per tonne primary Al",
    },
    IndustrySector.GLASS_FLAT: {
        "bat_ael_min_kwh": 1400, "bat_ael_max_kwh": 1600,
        "bref": "GLS BREF 2012", "bat_reference": "BAT 4",
        "notes": "Float glass process, per tonne molten glass",
    },
    IndustrySector.GLASS_CONTAINER: {
        "bat_ael_min_kwh": 1000, "bat_ael_max_kwh": 1200,
        "bref": "GLS BREF 2012", "bat_reference": "BAT 4",
        "notes": "Container glass, per tonne molten glass",
    },
    IndustrySector.CERAMICS: {
        "bat_ael_min_kwh": 500, "bat_ael_max_kwh": 600,
        "bref": "CER BREF 2007", "bat_reference": "BAT 12",
        "notes": "Wall and floor tiles, per tonne product",
    },
    IndustrySector.PULP_PAPER_INTEGRATED: {
        "bat_ael_min_kwh": 2500, "bat_ael_max_kwh": 3000,
        "bref": "PP BREF 2015", "bat_reference": "BAT 8",
        "notes": "Integrated kraft pulp and paper, per tonne product",
    },
    IndustrySector.FOOD_DAIRY: {
        "bat_ael_min_kwh": 150, "bat_ael_max_kwh": 200,
        "bref": "FDM BREF 2006", "bat_reference": "BAT 5",
        "notes": "Dairy processing, per tonne product",
    },
    IndustrySector.CHEMICALS_BULK: {
        "bat_ael_min_kwh": 1200, "bat_ael_max_kwh": 1500,
        "bref": "LVOC BREF 2017", "bat_reference": "BAT 5",
        "notes": "Bulk organic chemicals, per tonne product",
    },
    IndustrySector.PHARMACEUTICALS: {
        "bat_ael_min_kwh": 4500, "bat_ael_max_kwh": 6000,
        "bref": "OFC BREF 2006", "bat_reference": "BAT 3",
        "notes": "Active pharmaceutical ingredient manufacturing",
    },
}

# Energy rating thresholds.
# Rating is determined by the ratio of facility SEC to sector average SEC.
# A = <= 60% of average (top 10%), G = > 160% of average (worst 5%).
ENERGY_RATING_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    EnergyRatingClass.A: {
        "max_ratio": 0.60, "percentile_min": 90,
        "description": "Excellent - top decile energy performance",
        "colour": "#00B050",
    },
    EnergyRatingClass.B: {
        "max_ratio": 0.75, "percentile_min": 75,
        "description": "Very good - upper quartile performance",
        "colour": "#92D050",
    },
    EnergyRatingClass.C: {
        "max_ratio": 0.90, "percentile_min": 60,
        "description": "Good - above average performance",
        "colour": "#FFFF00",
    },
    EnergyRatingClass.D: {
        "max_ratio": 1.05, "percentile_min": 40,
        "description": "Average - median sector performance",
        "colour": "#FFC000",
    },
    EnergyRatingClass.E: {
        "max_ratio": 1.20, "percentile_min": 25,
        "description": "Below average - lower quartile",
        "colour": "#FF8000",
    },
    EnergyRatingClass.F: {
        "max_ratio": 1.60, "percentile_min": 10,
        "description": "Poor - significant improvement needed",
        "colour": "#FF4000",
    },
    EnergyRatingClass.G: {
        "max_ratio": 999.0, "percentile_min": 0,
        "description": "Very poor - urgent action required",
        "colour": "#FF0000",
    },
}

# Carbon intensity factors by energy carrier and country/region.
# Values in kgCO2/kWh.  Source: IEA, DEFRA 2024, EU ETS benchmarks.
CARBON_INTENSITY_FACTORS: Dict[str, Dict[str, float]] = {
    "electricity": {
        "EU_average": 0.250,
        "DE": 0.350,
        "FR": 0.055,
        "PL": 0.680,
        "ES": 0.180,
        "IT": 0.280,
        "NL": 0.340,
        "SE": 0.020,
        "UK": 0.230,
        "US": 0.380,
        "CN": 0.550,
        "IN": 0.720,
        "JP": 0.480,
        "KR": 0.460,
        "BR": 0.070,
        "global_average": 0.420,
    },
    "natural_gas": {
        "EU_average": 0.202,
        "global_average": 0.202,
    },
    "coal": {
        "EU_average": 0.340,
        "global_average": 0.340,
    },
    "fuel_oil": {
        "EU_average": 0.267,
        "global_average": 0.267,
    },
    "biomass": {
        "EU_average": 0.015,
        "global_average": 0.015,
    },
    "district_heat": {
        "EU_average": 0.180,
        "DE": 0.220,
        "SE": 0.060,
        "DK": 0.100,
        "global_average": 0.200,
    },
    "steam": {
        "EU_average": 0.200,
        "global_average": 0.200,
    },
}

# Energy cost reference by carrier (EUR/kWh) for savings estimation.
ENERGY_COST_REFERENCE: Dict[str, float] = {
    "electricity": 0.12,
    "natural_gas": 0.04,
    "coal": 0.02,
    "fuel_oil": 0.05,
    "biomass": 0.03,
    "district_heat": 0.06,
    "steam": 0.07,
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class BenchmarkFacility(BaseModel):
    """Facility data for energy benchmarking.

    Attributes:
        facility_id: Unique facility identifier.
        facility_name: Facility name.
        sector: Industrial sector classification.
        sub_sector: More specific sub-sector if applicable.
        production_output: Annual production volume.
        production_unit: Unit of production (tonnes, units, etc.).
        energy_consumption_kwh: Total annual energy consumption (kWh).
        energy_by_carrier: Breakdown by energy carrier (kWh).
        area_sqm: Facility floor area (m2).
        employees: Number of employees.
        revenue_eur: Annual revenue (EUR).
        country: ISO 2-letter country code.
        reporting_year: Year of the data.
        historical_sec: Previous years' SEC values for trajectory.
    """
    facility_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    sector: IndustrySector = Field(default=IndustrySector.GENERIC_MANUFACTURING)
    sub_sector: str = Field(default="")
    production_output: float = Field(default=0.0, ge=0.0)
    production_unit: str = Field(default="tonnes")
    energy_consumption_kwh: float = Field(default=0.0, ge=0.0)
    energy_by_carrier: Dict[str, float] = Field(
        default_factory=dict,
        description="Energy by carrier (kWh): electricity, natural_gas, etc."
    )
    area_sqm: float = Field(default=0.0, ge=0.0)
    employees: int = Field(default=0, ge=0)
    revenue_eur: float = Field(default=0.0, ge=0.0)
    country: str = Field(default="EU_average")
    reporting_year: int = Field(default=2025, ge=2015, le=2035)
    historical_sec: Dict[int, float] = Field(
        default_factory=dict,
        description="Historical SEC values: {year: sec_value}",
    )

class SECResult(BaseModel):
    """Specific Energy Consumption result.

    Attributes:
        sec_value: Facility SEC value.
        sec_unit: Unit of SEC.
        sector_best_practice: Best-practice SEC for the sector.
        sector_good: Good practice SEC.
        sector_average: Sector average SEC.
        sector_poor: Poor practice SEC.
        percentile_rank: Estimated percentile ranking (0-100).
        gap_to_best_pct: Gap to best practice (%).
        gap_to_best_kwh_per_unit: Absolute gap (kWh/unit).
        rating: Performance status string.
    """
    sec_value: float = Field(default=0.0)
    sec_unit: str = Field(default="kWh/tonne")
    sector_best_practice: float = Field(default=0.0)
    sector_good: float = Field(default=0.0)
    sector_average: float = Field(default=0.0)
    sector_poor: float = Field(default=0.0)
    percentile_rank: float = Field(default=0.0, ge=0.0, le=100.0)
    gap_to_best_pct: float = Field(default=0.0)
    gap_to_best_kwh_per_unit: float = Field(default=0.0)
    rating: str = Field(default="unknown")

class BATBenchmark(BaseModel):
    """BAT-AEL benchmark comparison result.

    Attributes:
        bref_document: Source BREF document identifier.
        bat_reference: Specific BAT conclusion reference.
        bat_ael_min: Lower bound of BAT-AEL range (kWh/tonne).
        bat_ael_max: Upper bound of BAT-AEL range (kWh/tonne).
        facility_value: Facility SEC value (kWh/tonne).
        compliance_status: BAT compliance status.
        gap_pct: Gap above BAT-AEL upper bound (%).
        gap_kwh: Absolute gap (kWh/tonne).
        notes: BAT conclusion notes.
    """
    bref_document: str = Field(default="")
    bat_reference: str = Field(default="")
    bat_ael_min: float = Field(default=0.0)
    bat_ael_max: float = Field(default=0.0)
    facility_value: float = Field(default=0.0)
    compliance_status: str = Field(default="not_assessed")
    gap_pct: float = Field(default=0.0)
    gap_kwh: float = Field(default=0.0)
    notes: str = Field(default="")

class EnergyRating(BaseModel):
    """Energy performance rating (A-G scale).

    Attributes:
        rating_class: Letter rating (A-G).
        score: Numeric score (0-100).
        ratio_to_average: SEC / sector average SEC.
        description: Rating description.
        improvement_needed_pct: Improvement needed to reach next class.
    """
    rating_class: EnergyRatingClass = Field(default=EnergyRatingClass.D)
    score: float = Field(default=50.0)
    ratio_to_average: float = Field(default=1.0)
    description: str = Field(default="")
    improvement_needed_pct: float = Field(default=0.0)

class PeerComparison(BaseModel):
    """Peer group comparison result.

    Attributes:
        peer_group_definition: How the peer group is defined.
        peer_group_size: Number of facilities in the peer group.
        facility_rank: Facility rank within peer group.
        percentile: Percentile position.
        quartile: Quartile (Q1=best, Q4=worst).
        median_sec: Median SEC of peer group.
        mean_sec: Mean SEC of peer group.
        best_sec: Best SEC in peer group.
        worst_sec: Worst SEC in peer group.
    """
    peer_group_definition: str = Field(default="")
    peer_group_size: int = Field(default=0)
    facility_rank: int = Field(default=0)
    percentile: float = Field(default=0.0)
    quartile: int = Field(default=2)
    median_sec: float = Field(default=0.0)
    mean_sec: float = Field(default=0.0)
    best_sec: float = Field(default=0.0)
    worst_sec: float = Field(default=0.0)

class CarbonIntensityResult(BaseModel):
    """Carbon intensity benchmarking result.

    Attributes:
        total_co2_kg: Total CO2 emissions (kg).
        carbon_intensity_kg_per_unit: kg CO2 per production unit.
        carbon_intensity_kg_per_kwh: kg CO2 per kWh consumed.
        carrier_breakdown: CO2 by energy carrier.
        country_grid_factor: Grid emission factor used.
    """
    total_co2_kg: float = Field(default=0.0)
    carbon_intensity_kg_per_unit: float = Field(default=0.0)
    carbon_intensity_kg_per_kwh: float = Field(default=0.0)
    carrier_breakdown: Dict[str, float] = Field(default_factory=dict)
    country_grid_factor: float = Field(default=0.0)

class TrajectoryPoint(BaseModel):
    """Single data point in the improvement trajectory."""
    year: int = Field(default=2025)
    sec_value: float = Field(default=0.0)
    year_over_year_change_pct: float = Field(default=0.0)
    cumulative_improvement_pct: float = Field(default=0.0)

class EnergyBenchmarkResult(BaseModel):
    """Complete energy benchmarking result with full provenance.

    Attributes:
        result_id: Unique result identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        sector: Industrial sector.
        sec_results: Specific Energy Consumption analysis.
        bat_benchmark: BAT-AEL comparison (if available).
        energy_rating: A-G energy performance rating.
        peer_comparison: Peer group comparison.
        carbon_intensity: Carbon intensity analysis.
        improvement_potential_kwh: Total annual improvement (kWh).
        improvement_potential_eur: Total annual savings (EUR).
        improvement_potential_tco2: Total CO2 reduction potential (tonnes).
        trajectory: Improvement trajectory data.
        multi_dimensional_scores: Scores across multiple dimensions.
        recommendations: Prioritised recommendations.
        methodology_notes: Methodology and source notes.
        processing_time_ms: Computation time (ms).
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    sector: str = Field(default="")
    sec_results: Optional[SECResult] = Field(default=None)
    bat_benchmark: Optional[BATBenchmark] = Field(default=None)
    energy_rating: Optional[EnergyRating] = Field(default=None)
    peer_comparison: Optional[PeerComparison] = Field(default=None)
    carbon_intensity: Optional[CarbonIntensityResult] = Field(default=None)
    improvement_potential_kwh: float = Field(default=0.0)
    improvement_potential_eur: float = Field(default=0.0)
    improvement_potential_tco2: float = Field(default=0.0)
    trajectory: List[TrajectoryPoint] = Field(default_factory=list)
    multi_dimensional_scores: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EnergyBenchmarkEngine:
    """Zero-hallucination energy benchmarking engine.

    Benchmarks facility energy performance against industry peers,
    EU BAT-AEL standards, and assigns A-G energy performance ratings.

    Guarantees:
        - Deterministic: same inputs produce identical outputs.
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full breakdown of SEC, BAT, rating, and trajectory.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = EnergyBenchmarkEngine()
        result = engine.benchmark(facility_data)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the energy benchmark engine.

        Args:
            config: Optional configuration overrides.
        """
        self._config = config or {}
        self._notes: List[str] = []
        logger.info("EnergyBenchmarkEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def benchmark(self, facility: BenchmarkFacility) -> EnergyBenchmarkResult:
        """Run comprehensive energy benchmarking for a facility.

        Calculates SEC, compares against BAT-AEL, assigns energy rating,
        estimates improvement potential, and builds improvement trajectory.

        Args:
            facility: Facility data for benchmarking.

        Returns:
            EnergyBenchmarkResult with full analysis and provenance.

        Raises:
            ValueError: If critical data is missing (e.g. zero energy).
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Sector: {facility.sector.value}",
            f"Reporting year: {facility.reporting_year}",
        ]

        if facility.energy_consumption_kwh <= 0:
            raise ValueError("Energy consumption must be > 0 for benchmarking.")

        recommendations: List[Dict[str, Any]] = []
        priority = 1

        # --- 1. SEC Calculation ---
        sec_result = self.calculate_sec(facility)

        # --- 2. BAT-AEL Comparison ---
        bat_result = self.compare_bat_ael(facility, sec_result)

        # --- 3. Energy Rating ---
        energy_rating = self.assign_energy_rating(sec_result)

        # --- 4. Peer Comparison ---
        peer_result = self.estimate_peer_comparison(sec_result, facility.sector)

        # --- 5. Carbon Intensity ---
        carbon_result = self.calculate_carbon_intensity(facility)

        # --- 6. Improvement Potential ---
        improvement_kwh, improvement_eur, improvement_tco2 = self._calculate_improvement(
            facility, sec_result, carbon_result
        )

        # --- 7. Trajectory ---
        trajectory = self.build_trajectory(facility)

        # --- 8. Multi-dimensional scores ---
        multi_scores = self._calculate_multi_dimensional(
            facility, sec_result, energy_rating, carbon_result
        )

        # --- 9. Recommendations ---
        if sec_result.gap_to_best_pct > 0:
            recommendations.append({
                "priority": priority,
                "category": "sec_improvement",
                "description": (
                    f"Close {_round1(sec_result.gap_to_best_pct)}% gap to sector "
                    f"best practice ({sec_result.sector_best_practice} {sec_result.sec_unit})."
                ),
                "savings_kwh": improvement_kwh,
                "savings_eur": improvement_eur,
                "co2_savings_tonnes": improvement_tco2,
            })
            priority += 1

        if bat_result and bat_result.compliance_status == "non_compliant":
            recommendations.append({
                "priority": priority,
                "category": "bat_compliance",
                "description": (
                    f"Achieve BAT-AEL compliance ({bat_result.bat_reference}: "
                    f"{bat_result.bat_ael_max} {sec_result.sec_unit})."
                ),
                "gap_kwh_per_unit": bat_result.gap_kwh,
                "gap_pct": bat_result.gap_pct,
            })
            priority += 1

        if energy_rating and energy_rating.rating_class in (
            EnergyRatingClass.E, EnergyRatingClass.F, EnergyRatingClass.G
        ):
            recommendations.append({
                "priority": priority,
                "category": "energy_rating_upgrade",
                "description": (
                    f"Upgrade energy rating from {energy_rating.rating_class.value} "
                    f"to at least D (average). Requires {_round1(energy_rating.improvement_needed_pct)}% improvement."
                ),
            })
            priority += 1

        if carbon_result and carbon_result.carbon_intensity_kg_per_kwh > 0.3:
            recommendations.append({
                "priority": priority,
                "category": "decarbonise_energy_mix",
                "description": (
                    f"High carbon intensity ({_round3(carbon_result.carbon_intensity_kg_per_kwh)} "
                    f"kgCO2/kWh). Shift to lower-carbon energy carriers."
                ),
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = EnergyBenchmarkResult(
            facility_id=facility.facility_id,
            facility_name=facility.facility_name,
            sector=facility.sector.value,
            sec_results=sec_result,
            bat_benchmark=bat_result,
            energy_rating=energy_rating,
            peer_comparison=peer_result,
            carbon_intensity=carbon_result,
            improvement_potential_kwh=_round2(improvement_kwh),
            improvement_potential_eur=_round2(improvement_eur),
            improvement_potential_tco2=_round3(improvement_tco2),
            trajectory=trajectory,
            multi_dimensional_scores=multi_scores,
            recommendations=recommendations,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    # --------------------------------------------------------------------- #
    # SEC Calculation
    # --------------------------------------------------------------------- #

    def calculate_sec(self, facility: BenchmarkFacility) -> SECResult:
        """Calculate Specific Energy Consumption (SEC).

        SEC = Total Energy (kWh) / Production Output

        Also performs percentile estimation and gap analysis against
        sector benchmarks.

        Args:
            facility: Facility data.

        Returns:
            SECResult with benchmarking metrics.
        """
        d_energy = _decimal(facility.energy_consumption_kwh)
        d_output = _decimal(facility.production_output)

        sec_value = _safe_divide(d_energy, d_output) if d_output > Decimal("0") else Decimal("0")

        # Look up sector benchmarks.
        benchmarks = SEC_BENCHMARKS.get(facility.sector, SEC_BENCHMARKS[IndustrySector.GENERIC_MANUFACTURING])
        d_best = _decimal(benchmarks["best_practice"])
        d_good = _decimal(benchmarks["good"])
        d_avg = _decimal(benchmarks["average"])
        d_poor = _decimal(benchmarks["poor"])
        sec_unit = benchmarks.get("unit", "kWh/tonne")

        # Percentile ranking (piecewise linear interpolation).
        percentile = self._estimate_percentile(sec_value, d_best, d_good, d_avg, d_poor)

        # Gap to best practice.
        gap_to_best_pct = Decimal("0")
        gap_to_best_kwh = Decimal("0")
        if sec_value > d_best and d_best > Decimal("0"):
            gap_to_best_kwh = sec_value - d_best
            gap_to_best_pct = _safe_divide(gap_to_best_kwh, d_best) * Decimal("100")

        # Rating.
        if sec_value <= d_best:
            rating = "best_practice"
        elif sec_value <= d_good:
            rating = "good"
        elif sec_value <= d_avg:
            rating = "average"
        else:
            rating = "poor"

        self._notes.append(
            f"SEC: {_round2(float(sec_value))} {sec_unit}, "
            f"sector best {float(d_best)}, average {float(d_avg)}, "
            f"percentile rank {_round1(float(percentile))}."
        )

        return SECResult(
            sec_value=_round2(float(sec_value)),
            sec_unit=sec_unit,
            sector_best_practice=float(d_best),
            sector_good=float(d_good),
            sector_average=float(d_avg),
            sector_poor=float(d_poor),
            percentile_rank=_round1(float(percentile)),
            gap_to_best_pct=_round2(float(gap_to_best_pct)),
            gap_to_best_kwh_per_unit=_round2(float(gap_to_best_kwh)),
            rating=rating,
        )

    # --------------------------------------------------------------------- #
    # BAT-AEL Comparison
    # --------------------------------------------------------------------- #

    def compare_bat_ael(
        self,
        facility: BenchmarkFacility,
        sec_result: SECResult,
    ) -> Optional[BATBenchmark]:
        """Compare facility SEC against BAT-AEL from BREF documents.

        Args:
            facility: Facility data.
            sec_result: Previously calculated SEC.

        Returns:
            BATBenchmark if BAT-AEL data exists for the sector, else None.
        """
        bat_data = BAT_AEL_DATABASE.get(facility.sector)
        if not bat_data:
            self._notes.append(f"No BAT-AEL data available for sector {facility.sector.value}.")
            return None

        d_sec = _decimal(sec_result.sec_value)
        d_min = _decimal(bat_data["bat_ael_min_kwh"])
        d_max = _decimal(bat_data["bat_ael_max_kwh"])

        if d_sec <= d_min:
            status = "compliant_below_range"
            gap_pct = Decimal("0")
            gap_kwh = Decimal("0")
        elif d_sec <= d_max:
            status = "within_range"
            gap_pct = Decimal("0")
            gap_kwh = Decimal("0")
        else:
            status = "non_compliant"
            gap_kwh = d_sec - d_max
            gap_pct = _safe_divide(gap_kwh, d_max) * Decimal("100")

        self._notes.append(
            f"BAT-AEL ({bat_data['bref']} {bat_data['bat_reference']}): "
            f"range {float(d_min)}-{float(d_max)}, facility {float(d_sec)}, "
            f"status: {status}."
        )

        return BATBenchmark(
            bref_document=bat_data["bref"],
            bat_reference=bat_data["bat_reference"],
            bat_ael_min=float(d_min),
            bat_ael_max=float(d_max),
            facility_value=sec_result.sec_value,
            compliance_status=status,
            gap_pct=_round2(float(gap_pct)),
            gap_kwh=_round2(float(gap_kwh)),
            notes=bat_data.get("notes", ""),
        )

    # --------------------------------------------------------------------- #
    # Energy Rating (A-G)
    # --------------------------------------------------------------------- #

    def assign_energy_rating(self, sec_result: SECResult) -> EnergyRating:
        """Assign an A-G energy performance rating.

        Rating is based on the ratio of facility SEC to sector average SEC.

        Args:
            sec_result: SEC analysis result.

        Returns:
            EnergyRating with class, score, and improvement target.
        """
        d_sec = _decimal(sec_result.sec_value)
        d_avg = _decimal(sec_result.sector_average)

        ratio = _safe_divide(d_sec, d_avg, default=Decimal("1.0"))

        assigned_class = EnergyRatingClass.G
        for cls in [EnergyRatingClass.A, EnergyRatingClass.B, EnergyRatingClass.C,
                     EnergyRatingClass.D, EnergyRatingClass.E, EnergyRatingClass.F,
                     EnergyRatingClass.G]:
            threshold = _decimal(ENERGY_RATING_THRESHOLDS[cls]["max_ratio"])
            if ratio <= threshold:
                assigned_class = cls
                break

        # Score: inverse of ratio, scaled 0-100.
        # A (ratio 0.60) = 100, G (ratio 1.60+) = 0.
        score = max(Decimal("0"), (Decimal("1.60") - ratio) / Decimal("1.00") * Decimal("100"))
        score = min(score, Decimal("100"))

        # Improvement needed to reach next better class.
        improvement_pct = Decimal("0")
        rating_order = [EnergyRatingClass.A, EnergyRatingClass.B, EnergyRatingClass.C,
                        EnergyRatingClass.D, EnergyRatingClass.E, EnergyRatingClass.F,
                        EnergyRatingClass.G]
        current_idx = rating_order.index(assigned_class)
        if current_idx > 0:
            better_class = rating_order[current_idx - 1]
            target_ratio = _decimal(ENERGY_RATING_THRESHOLDS[better_class]["max_ratio"])
            if ratio > target_ratio:
                improvement_pct = (Decimal("1") - _safe_divide(target_ratio, ratio)) * Decimal("100")

        description = ENERGY_RATING_THRESHOLDS[assigned_class]["description"]

        self._notes.append(
            f"Energy rating: {assigned_class.value} (ratio to average: "
            f"{_round2(float(ratio))}, score: {_round1(float(score))})."
        )

        return EnergyRating(
            rating_class=assigned_class,
            score=_round1(float(score)),
            ratio_to_average=_round3(float(ratio)),
            description=description,
            improvement_needed_pct=_round2(float(improvement_pct)),
        )

    # --------------------------------------------------------------------- #
    # Peer Comparison
    # --------------------------------------------------------------------- #

    def estimate_peer_comparison(
        self,
        sec_result: SECResult,
        sector: IndustrySector,
    ) -> PeerComparison:
        """Estimate peer group comparison from sector benchmark data.

        Since we do not have access to a full peer database, the peer
        group is synthesised from sector benchmarks (best, good, average,
        poor) to provide a representative distribution.

        Args:
            sec_result: SEC analysis result.
            sector: Industrial sector.

        Returns:
            PeerComparison with estimated ranking.
        """
        d_sec = _decimal(sec_result.sec_value)
        d_best = _decimal(sec_result.sector_best_practice)
        d_good = _decimal(sec_result.sector_good)
        d_avg = _decimal(sec_result.sector_average)
        d_poor = _decimal(sec_result.sector_poor)

        # Estimate peer group of 100 hypothetical facilities.
        peer_size = 100
        percentile = _decimal(sec_result.percentile_rank)
        rank = max(1, int(float((Decimal("100") - percentile) / Decimal("100") * _decimal(peer_size))))

        # Quartile assignment.
        if percentile >= Decimal("75"):
            quartile = 1
        elif percentile >= Decimal("50"):
            quartile = 2
        elif percentile >= Decimal("25"):
            quartile = 3
        else:
            quartile = 4

        # Synthesised peer group stats.
        median_sec = d_avg  # Median ~ average in a normal-like distribution.
        mean_sec = (d_best + d_good + d_avg + d_poor) / Decimal("4")

        definition = f"Sector: {sector.value}, estimated from BREF/IEA benchmark data"

        return PeerComparison(
            peer_group_definition=definition,
            peer_group_size=peer_size,
            facility_rank=rank,
            percentile=_round1(float(percentile)),
            quartile=quartile,
            median_sec=_round2(float(median_sec)),
            mean_sec=_round2(float(mean_sec)),
            best_sec=_round2(float(d_best)),
            worst_sec=_round2(float(d_poor)),
        )

    # --------------------------------------------------------------------- #
    # Carbon Intensity
    # --------------------------------------------------------------------- #

    def calculate_carbon_intensity(
        self,
        facility: BenchmarkFacility,
    ) -> CarbonIntensityResult:
        """Calculate carbon intensity alongside energy intensity.

        Uses country-specific grid emission factors and carrier-specific
        CO2 factors to compute total emissions and carbon intensity per
        production unit.

        Args:
            facility: Facility data with energy-by-carrier breakdown.

        Returns:
            CarbonIntensityResult with emission breakdown.
        """
        total_co2_kg = Decimal("0")
        carrier_breakdown: Dict[str, float] = {}

        country = facility.country if facility.country else "EU_average"

        if facility.energy_by_carrier:
            for carrier, kwh in facility.energy_by_carrier.items():
                d_kwh = _decimal(kwh)
                factor = self._get_carbon_factor(carrier, country)
                co2_kg = d_kwh * factor
                total_co2_kg += co2_kg
                carrier_breakdown[carrier] = _round2(float(co2_kg))
        else:
            # Assume all electricity if no breakdown provided.
            d_energy = _decimal(facility.energy_consumption_kwh)
            factor = self._get_carbon_factor("electricity", country)
            total_co2_kg = d_energy * factor
            carrier_breakdown["electricity"] = _round2(float(total_co2_kg))

        d_output = _decimal(facility.production_output)
        d_energy = _decimal(facility.energy_consumption_kwh)

        carbon_per_unit = _safe_divide(total_co2_kg, d_output) if d_output > Decimal("0") else Decimal("0")
        carbon_per_kwh = _safe_divide(total_co2_kg, d_energy) if d_energy > Decimal("0") else Decimal("0")

        grid_factor = float(self._get_carbon_factor("electricity", country))

        self._notes.append(
            f"Carbon intensity: {_round3(float(carbon_per_unit))} kgCO2/{facility.production_unit}, "
            f"total {_round2(float(total_co2_kg / Decimal('1000')))} tCO2, "
            f"grid factor {grid_factor} kgCO2/kWh ({country})."
        )

        return CarbonIntensityResult(
            total_co2_kg=_round2(float(total_co2_kg)),
            carbon_intensity_kg_per_unit=_round3(float(carbon_per_unit)),
            carbon_intensity_kg_per_kwh=_round3(float(carbon_per_kwh)),
            carrier_breakdown=carrier_breakdown,
            country_grid_factor=grid_factor,
        )

    # --------------------------------------------------------------------- #
    # Improvement Trajectory
    # --------------------------------------------------------------------- #

    def build_trajectory(self, facility: BenchmarkFacility) -> List[TrajectoryPoint]:
        """Build improvement trajectory from historical SEC data.

        Calculates year-over-year and cumulative improvement from
        historical SEC values provided.

        Args:
            facility: Facility data with historical_sec dictionary.

        Returns:
            List of TrajectoryPoint sorted by year.
        """
        if not facility.historical_sec:
            return []

        # Sort by year.
        sorted_years = sorted(facility.historical_sec.keys())
        points: List[TrajectoryPoint] = []
        baseline_sec = _decimal(facility.historical_sec[sorted_years[0]])

        prev_sec = baseline_sec
        for year in sorted_years:
            d_sec = _decimal(facility.historical_sec[year])
            yoy_change = Decimal("0")
            if prev_sec > Decimal("0"):
                yoy_change = _safe_divide(d_sec - prev_sec, prev_sec) * Decimal("100")

            cumulative = Decimal("0")
            if baseline_sec > Decimal("0"):
                cumulative = _safe_divide(baseline_sec - d_sec, baseline_sec) * Decimal("100")

            points.append(TrajectoryPoint(
                year=year,
                sec_value=_round2(float(d_sec)),
                year_over_year_change_pct=_round2(float(yoy_change)),
                cumulative_improvement_pct=_round2(float(cumulative)),
            ))
            prev_sec = d_sec

        if points:
            self._notes.append(
                f"Trajectory: {len(points)} data points, baseline {float(baseline_sec)} "
                f"({sorted_years[0]}), latest {float(prev_sec)} ({sorted_years[-1]}), "
                f"cumulative improvement {points[-1].cumulative_improvement_pct}%."
            )

        return points

    # --------------------------------------------------------------------- #
    # Private Helpers
    # --------------------------------------------------------------------- #

    def _estimate_percentile(
        self,
        sec_value: Decimal,
        best: Decimal,
        good: Decimal,
        average: Decimal,
        poor: Decimal,
    ) -> Decimal:
        """Estimate percentile rank using piecewise linear interpolation.

        Mapping:
            SEC <= best    -> 95th percentile (top 5%)
            SEC = good     -> 75th percentile
            SEC = average  -> 50th percentile
            SEC = poor     -> 15th percentile
            SEC > poor     -> 5th percentile

        Lower SEC is better, so higher percentile = better.

        Args:
            sec_value: Facility SEC.
            best: Best-practice SEC.
            good: Good practice SEC.
            average: Sector average SEC.
            poor: Poor practice SEC.

        Returns:
            Percentile as Decimal (0-100).
        """
        if sec_value <= best:
            return Decimal("95")
        elif sec_value <= good:
            # Interpolate 95 -> 75 as SEC goes from best -> good.
            fraction = _safe_divide(sec_value - best, good - best)
            return Decimal("95") - fraction * Decimal("20")
        elif sec_value <= average:
            # Interpolate 75 -> 50.
            fraction = _safe_divide(sec_value - good, average - good)
            return Decimal("75") - fraction * Decimal("25")
        elif sec_value <= poor:
            # Interpolate 50 -> 15.
            fraction = _safe_divide(sec_value - average, poor - average)
            return Decimal("50") - fraction * Decimal("35")
        else:
            # Below poor: 15 -> 5.
            excess = poor * Decimal("0.5")  # Assume worst = 1.5x poor.
            if excess > Decimal("0"):
                fraction = min(
                    _safe_divide(sec_value - poor, excess),
                    Decimal("1"),
                )
                return Decimal("15") - fraction * Decimal("10")
            return Decimal("5")

    def _calculate_improvement(
        self,
        facility: BenchmarkFacility,
        sec_result: SECResult,
        carbon_result: CarbonIntensityResult,
    ) -> Tuple[float, float, float]:
        """Calculate total improvement potential.

        Improvement = gap to good practice SEC * production volume.

        Args:
            facility: Facility data.
            sec_result: SEC results.
            carbon_result: Carbon intensity results.

        Returns:
            Tuple of (improvement_kwh, improvement_eur, improvement_tco2).
        """
        d_sec = _decimal(sec_result.sec_value)
        d_target = _decimal(sec_result.sector_good)
        d_output = _decimal(facility.production_output)

        gap_kwh_per_unit = max(d_sec - d_target, Decimal("0"))
        improvement_kwh = gap_kwh_per_unit * d_output

        # Cost savings: use weighted average energy cost.
        avg_cost = Decimal("0")
        if facility.energy_by_carrier:
            total_kwh = Decimal("0")
            total_cost = Decimal("0")
            for carrier, kwh in facility.energy_by_carrier.items():
                d_kwh = _decimal(kwh)
                cost_ref = _decimal(ENERGY_COST_REFERENCE.get(carrier, 0.08))
                total_kwh += d_kwh
                total_cost += d_kwh * cost_ref
            avg_cost = _safe_divide(total_cost, total_kwh) if total_kwh > Decimal("0") else Decimal("0.08")
        else:
            avg_cost = Decimal("0.10")

        improvement_eur = improvement_kwh * avg_cost

        # CO2 reduction.
        d_ci = _decimal(carbon_result.carbon_intensity_kg_per_kwh) if carbon_result else Decimal("0.25")
        improvement_co2_kg = improvement_kwh * d_ci
        improvement_tco2 = improvement_co2_kg / Decimal("1000")

        return (
            _round2(float(improvement_kwh)),
            _round2(float(improvement_eur)),
            _round3(float(improvement_tco2)),
        )

    def _calculate_multi_dimensional(
        self,
        facility: BenchmarkFacility,
        sec_result: SECResult,
        energy_rating: EnergyRating,
        carbon_result: CarbonIntensityResult,
    ) -> Dict[str, Any]:
        """Calculate multi-dimensional benchmark scores.

        Provides scores across several dimensions:
        - Energy intensity (SEC)
        - Carbon intensity
        - Energy productivity (EUR revenue per kWh)
        - Floor area intensity (kWh per m2)
        - Employee productivity (kWh per employee)

        Args:
            facility: Facility data.
            sec_result: SEC analysis.
            energy_rating: Energy rating.
            carbon_result: Carbon intensity.

        Returns:
            Dictionary of multi-dimensional scores.
        """
        d_energy = _decimal(facility.energy_consumption_kwh)
        scores: Dict[str, Any] = {}

        # Energy intensity score (from rating).
        scores["energy_intensity"] = {
            "value": sec_result.sec_value,
            "unit": sec_result.sec_unit,
            "percentile": sec_result.percentile_rank,
            "rating": energy_rating.rating_class.value if energy_rating else "N/A",
        }

        # Carbon intensity.
        if carbon_result:
            scores["carbon_intensity"] = {
                "value": carbon_result.carbon_intensity_kg_per_unit,
                "unit": f"kgCO2/{facility.production_unit}",
                "total_tco2": _round2(float(_decimal(carbon_result.total_co2_kg) / Decimal("1000"))),
            }

        # Energy productivity (revenue per kWh).
        if facility.revenue_eur > 0 and d_energy > Decimal("0"):
            productivity = _safe_divide(_decimal(facility.revenue_eur), d_energy)
            scores["energy_productivity"] = {
                "value": _round3(float(productivity)),
                "unit": "EUR/kWh",
                "description": "Revenue generated per kWh consumed",
            }

        # Floor area intensity (EUI).
        if facility.area_sqm > 0 and d_energy > Decimal("0"):
            eui = _safe_divide(d_energy, _decimal(facility.area_sqm))
            scores["floor_area_intensity"] = {
                "value": _round2(float(eui)),
                "unit": "kWh/m2/year",
                "description": "Energy Use Intensity per floor area",
            }

        # Employee intensity.
        if facility.employees > 0 and d_energy > Decimal("0"):
            per_employee = _safe_divide(d_energy, _decimal(facility.employees))
            scores["employee_intensity"] = {
                "value": _round2(float(per_employee)),
                "unit": "kWh/employee/year",
                "description": "Energy consumption per employee",
            }

        return scores

    def _get_carbon_factor(self, carrier: str, country: str) -> Decimal:
        """Look up carbon emission factor for an energy carrier and country.

        Falls back to EU_average, then global_average if country not found.

        Args:
            carrier: Energy carrier name.
            country: ISO country code or 'EU_average'.

        Returns:
            Emission factor in kgCO2/kWh as Decimal.
        """
        carrier_lower = carrier.lower()
        factors = CARBON_INTENSITY_FACTORS.get(carrier_lower, {})

        if country in factors:
            return _decimal(factors[country])
        elif "EU_average" in factors:
            return _decimal(factors["EU_average"])
        elif "global_average" in factors:
            return _decimal(factors["global_average"])
        else:
            return Decimal("0.25")  # Conservative default
