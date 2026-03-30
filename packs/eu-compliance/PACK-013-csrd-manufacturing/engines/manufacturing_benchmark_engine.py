# -*- coding: utf-8 -*-
"""
ManufacturingBenchmarkEngine - PACK-013 CSRD Manufacturing Engine 8
=====================================================================

Sector benchmarking engine for manufacturing sustainability performance.
Ranks facility KPIs against sector peers, assesses SBTi pathway alignment,
calculates EU ETS benchmark gaps, and performs trajectory analysis for
ESRS E1/E2/E3/E5 disclosure requirements.

Benchmarking Capabilities:
    - Emission intensity ranking (tCO2e per production unit)
    - Energy intensity ranking (MJ per production unit)
    - Water intensity ranking (m3 per production unit)
    - Waste intensity ranking (kg per production unit)
    - Circularity rate benchmarking
    - Renewable energy share benchmarking
    - Scope 3 ratio analysis
    - Safety performance (LTIR) benchmarking
    - SBTi pathway alignment (well-below 2C, 1.5C, net-zero)
    - EU ETS product benchmark gap analysis
    - Multi-year trajectory analysis

Regulatory References:
    - ESRS E1 (Climate Change) - E1-5/E1-6 transition plan KPIs
    - ESRS E2 (Pollution) - E2-4 intensity metrics
    - ESRS E3 (Water) - E3-4 water intensity
    - ESRS E5 (Circular Economy) - E5-5 resource efficiency
    - EU ETS Directive (2003/87/EC, amended)
    - SBTi Corporate Net-Zero Standard (v1.0)

Zero-Hallucination:
    - All benchmark data sourced from published sector reports
    - Percentile ranking uses deterministic comparison logic
    - SBTi pathway targets from official SBTi documentation
    - EU ETS benchmarks from Commission Delegated Regulation
    - SHA-256 provenance hash on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-013 CSRD Manufacturing
Status: Production Ready
"""

import hashlib
import json
import logging
import math
import time
import uuid
from collections import defaultdict
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

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _round_value(value: Decimal, places: int = 3) -> float:
    """Round a Decimal to specified places and return float."""
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BenchmarkKPI(str, Enum):
    """Key performance indicator for benchmarking."""
    EMISSION_INTENSITY = "emission_intensity"
    ENERGY_INTENSITY = "energy_intensity"
    WATER_INTENSITY = "water_intensity"
    WASTE_INTENSITY = "waste_intensity"
    CIRCULARITY_RATE = "circularity_rate"
    RENEWABLE_SHARE = "renewable_share"
    SCOPE3_RATIO = "scope3_ratio"
    SAFETY_LTIR = "safety_ltir"

class PercentileRank(str, Enum):
    """Quartile classification for benchmark ranking."""
    TOP_QUARTILE = "top_quartile"
    SECOND_QUARTILE = "second_quartile"
    THIRD_QUARTILE = "third_quartile"
    BOTTOM_QUARTILE = "bottom_quartile"

class SBTiPathway(str, Enum):
    """SBTi target-setting pathway."""
    WELL_BELOW_2C = "well_below_2c"
    ONE_POINT_FIVE_C = "one_point_five_c"
    NET_ZERO_2050 = "net_zero_2050"

class SubSector(str, Enum):
    """Manufacturing sub-sector classification."""
    CEMENT = "cement"
    STEEL_BOF = "steel_bof"
    STEEL_EAF = "steel_eaf"
    ALUMINIUM = "aluminium"
    GLASS = "glass"
    CERAMICS = "ceramics"
    CHEMICALS = "chemicals"
    CHEMICALS_SPECIALTY = "chemicals_specialty"
    PULP_PAPER = "pulp_paper"
    FOOD_DRINK = "food_drink"
    TEXTILES = "textiles"
    AUTOMOTIVE = "automotive"
    ELECTRONICS = "electronics"
    MACHINERY = "machinery"
    PHARMACEUTICALS = "pharmaceuticals"

# ---------------------------------------------------------------------------
# Constants - Sector Benchmarks
# ---------------------------------------------------------------------------

# Sector benchmark data compiled from:
# - IEA Energy Technology Perspectives (2024)
# - World Benchmarking Alliance (2024)
# - Transition Pathway Initiative (TPI)
# - EU ETS benchmarks (Commission Delegated Regulation)
# - CDP sector reports
#
# Format: {sub_sector: {kpi: {top_quartile, median, bottom_quartile, unit, source, year}}}

SECTOR_BENCHMARKS: Dict[str, Dict[str, Dict[str, Any]]] = {
    SubSector.CEMENT: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 0.55, "median": 0.63, "bottom_quartile": 0.78,
            "unit": "tCO2/t_clinker", "source": "GCCA/IEA 2024", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 3200.0, "median": 3500.0, "bottom_quartile": 3900.0,
            "unit": "MJ/t_clinker", "source": "IEA Cement Roadmap", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 0.20, "median": 0.35, "bottom_quartile": 0.55,
            "unit": "m3/t_cement", "source": "WBCSD CSI", "year": 2024,
        },
        BenchmarkKPI.WASTE_INTENSITY: {
            "top_quartile": 5.0, "median": 12.0, "bottom_quartile": 25.0,
            "unit": "kg/t_cement", "source": "Industry average", "year": 2024,
        },
        BenchmarkKPI.CIRCULARITY_RATE: {
            "top_quartile": 30.0, "median": 18.0, "bottom_quartile": 8.0,
            "unit": "%", "source": "GCCA", "year": 2024,
        },
        BenchmarkKPI.RENEWABLE_SHARE: {
            "top_quartile": 25.0, "median": 12.0, "bottom_quartile": 5.0,
            "unit": "%", "source": "Industry reports", "year": 2024,
        },
        BenchmarkKPI.SCOPE3_RATIO: {
            "top_quartile": 15.0, "median": 22.0, "bottom_quartile": 30.0,
            "unit": "%", "source": "CDP sector analysis", "year": 2024,
        },
        BenchmarkKPI.SAFETY_LTIR: {
            "top_quartile": 0.8, "median": 2.5, "bottom_quartile": 5.0,
            "unit": "per_million_hours", "source": "GCCA", "year": 2024,
        },
    },
    SubSector.STEEL_BOF: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 1.60, "median": 1.85, "bottom_quartile": 2.20,
            "unit": "tCO2/t_steel", "source": "World Steel Assoc.", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 18000.0, "median": 20500.0, "bottom_quartile": 24000.0,
            "unit": "MJ/t_steel", "source": "World Steel Assoc.", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 2.5, "median": 3.8, "bottom_quartile": 6.0,
            "unit": "m3/t_steel", "source": "World Steel Assoc.", "year": 2024,
        },
        BenchmarkKPI.WASTE_INTENSITY: {
            "top_quartile": 200.0, "median": 350.0, "bottom_quartile": 500.0,
            "unit": "kg/t_steel", "source": "Industry average", "year": 2024,
        },
        BenchmarkKPI.CIRCULARITY_RATE: {
            "top_quartile": 95.0, "median": 85.0, "bottom_quartile": 70.0,
            "unit": "%", "source": "BIR/World Steel", "year": 2024,
        },
        BenchmarkKPI.RENEWABLE_SHARE: {
            "top_quartile": 30.0, "median": 15.0, "bottom_quartile": 5.0,
            "unit": "%", "source": "Industry reports", "year": 2024,
        },
        BenchmarkKPI.SCOPE3_RATIO: {
            "top_quartile": 20.0, "median": 30.0, "bottom_quartile": 45.0,
            "unit": "%", "source": "CDP sector analysis", "year": 2024,
        },
        BenchmarkKPI.SAFETY_LTIR: {
            "top_quartile": 0.5, "median": 1.8, "bottom_quartile": 4.0,
            "unit": "per_million_hours", "source": "World Steel", "year": 2024,
        },
    },
    SubSector.STEEL_EAF: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 0.30, "median": 0.40, "bottom_quartile": 0.60,
            "unit": "tCO2/t_steel", "source": "World Steel Assoc.", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 4500.0, "median": 5500.0, "bottom_quartile": 7000.0,
            "unit": "MJ/t_steel", "source": "World Steel Assoc.", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 1.0, "median": 1.8, "bottom_quartile": 3.0,
            "unit": "m3/t_steel", "source": "World Steel Assoc.", "year": 2024,
        },
        BenchmarkKPI.WASTE_INTENSITY: {
            "top_quartile": 100.0, "median": 180.0, "bottom_quartile": 300.0,
            "unit": "kg/t_steel", "source": "Industry average", "year": 2024,
        },
        BenchmarkKPI.CIRCULARITY_RATE: {
            "top_quartile": 98.0, "median": 92.0, "bottom_quartile": 80.0,
            "unit": "%", "source": "BIR/World Steel", "year": 2024,
        },
        BenchmarkKPI.RENEWABLE_SHARE: {
            "top_quartile": 60.0, "median": 35.0, "bottom_quartile": 15.0,
            "unit": "%", "source": "Industry reports", "year": 2024,
        },
        BenchmarkKPI.SCOPE3_RATIO: {
            "top_quartile": 35.0, "median": 50.0, "bottom_quartile": 65.0,
            "unit": "%", "source": "CDP sector analysis", "year": 2024,
        },
        BenchmarkKPI.SAFETY_LTIR: {
            "top_quartile": 0.5, "median": 1.5, "bottom_quartile": 3.5,
            "unit": "per_million_hours", "source": "World Steel", "year": 2024,
        },
    },
    SubSector.ALUMINIUM: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 4.0, "median": 8.0, "bottom_quartile": 12.5,
            "unit": "tCO2/t_aluminium", "source": "IAI", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 50000.0, "median": 58000.0, "bottom_quartile": 68000.0,
            "unit": "MJ/t_aluminium", "source": "IAI", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 2.0, "median": 3.5, "bottom_quartile": 6.0,
            "unit": "m3/t_aluminium", "source": "IAI", "year": 2024,
        },
        BenchmarkKPI.CIRCULARITY_RATE: {
            "top_quartile": 75.0, "median": 55.0, "bottom_quartile": 30.0,
            "unit": "%", "source": "IAI", "year": 2024,
        },
        BenchmarkKPI.RENEWABLE_SHARE: {
            "top_quartile": 80.0, "median": 50.0, "bottom_quartile": 20.0,
            "unit": "%", "source": "IAI", "year": 2024,
        },
    },
    SubSector.AUTOMOTIVE: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 0.50, "median": 0.80, "bottom_quartile": 1.20,
            "unit": "tCO2e/vehicle", "source": "ACEA/CDP", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 4500.0, "median": 6000.0, "bottom_quartile": 8500.0,
            "unit": "MJ/vehicle", "source": "Industry average", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 2.5, "median": 4.0, "bottom_quartile": 6.5,
            "unit": "m3/vehicle", "source": "Industry average", "year": 2024,
        },
        BenchmarkKPI.WASTE_INTENSITY: {
            "top_quartile": 8.0, "median": 15.0, "bottom_quartile": 30.0,
            "unit": "kg/vehicle", "source": "Industry average", "year": 2024,
        },
        BenchmarkKPI.RENEWABLE_SHARE: {
            "top_quartile": 60.0, "median": 35.0, "bottom_quartile": 12.0,
            "unit": "%", "source": "Industry reports", "year": 2024,
        },
        BenchmarkKPI.SCOPE3_RATIO: {
            "top_quartile": 70.0, "median": 80.0, "bottom_quartile": 90.0,
            "unit": "%", "source": "CDP sector analysis", "year": 2024,
        },
    },
    SubSector.CHEMICALS: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 0.80, "median": 1.20, "bottom_quartile": 1.80,
            "unit": "tCO2e/t_product", "source": "Cefic/IEA", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 8000.0, "median": 12000.0, "bottom_quartile": 18000.0,
            "unit": "MJ/t_product", "source": "Cefic/IEA", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 3.0, "median": 6.0, "bottom_quartile": 12.0,
            "unit": "m3/t_product", "source": "Industry average", "year": 2024,
        },
        BenchmarkKPI.WASTE_INTENSITY: {
            "top_quartile": 20.0, "median": 50.0, "bottom_quartile": 100.0,
            "unit": "kg/t_product", "source": "Industry average", "year": 2024,
        },
        BenchmarkKPI.RENEWABLE_SHARE: {
            "top_quartile": 35.0, "median": 18.0, "bottom_quartile": 7.0,
            "unit": "%", "source": "Cefic", "year": 2024,
        },
        BenchmarkKPI.SCOPE3_RATIO: {
            "top_quartile": 40.0, "median": 55.0, "bottom_quartile": 70.0,
            "unit": "%", "source": "CDP sector analysis", "year": 2024,
        },
    },
    SubSector.GLASS: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 0.45, "median": 0.60, "bottom_quartile": 0.85,
            "unit": "tCO2/t_glass", "source": "Glass Alliance Europe", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 5000.0, "median": 6500.0, "bottom_quartile": 8500.0,
            "unit": "MJ/t_glass", "source": "Glass Alliance Europe", "year": 2024,
        },
        BenchmarkKPI.CIRCULARITY_RATE: {
            "top_quartile": 85.0, "median": 65.0, "bottom_quartile": 40.0,
            "unit": "%", "source": "FEVE", "year": 2024,
        },
    },
    SubSector.PULP_PAPER: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 0.25, "median": 0.40, "bottom_quartile": 0.65,
            "unit": "tCO2/t_product", "source": "CEPI", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 10000.0, "median": 14000.0, "bottom_quartile": 20000.0,
            "unit": "MJ/t_product", "source": "CEPI", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 15.0, "median": 25.0, "bottom_quartile": 45.0,
            "unit": "m3/t_product", "source": "CEPI", "year": 2024,
        },
        BenchmarkKPI.RENEWABLE_SHARE: {
            "top_quartile": 70.0, "median": 55.0, "bottom_quartile": 30.0,
            "unit": "%", "source": "CEPI", "year": 2024,
        },
    },
    SubSector.FOOD_DRINK: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 0.10, "median": 0.25, "bottom_quartile": 0.50,
            "unit": "tCO2e/t_product", "source": "FoodDrinkEurope", "year": 2024,
        },
        BenchmarkKPI.ENERGY_INTENSITY: {
            "top_quartile": 1500.0, "median": 2500.0, "bottom_quartile": 4500.0,
            "unit": "MJ/t_product", "source": "FoodDrinkEurope", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 2.0, "median": 5.0, "bottom_quartile": 12.0,
            "unit": "m3/t_product", "source": "FoodDrinkEurope", "year": 2024,
        },
        BenchmarkKPI.WASTE_INTENSITY: {
            "top_quartile": 10.0, "median": 30.0, "bottom_quartile": 60.0,
            "unit": "kg/t_product", "source": "Industry average", "year": 2024,
        },
    },
    SubSector.TEXTILES: {
        BenchmarkKPI.EMISSION_INTENSITY: {
            "top_quartile": 5.0, "median": 10.0, "bottom_quartile": 18.0,
            "unit": "tCO2e/t_textile", "source": "Textile Exchange", "year": 2024,
        },
        BenchmarkKPI.WATER_INTENSITY: {
            "top_quartile": 30.0, "median": 80.0, "bottom_quartile": 200.0,
            "unit": "m3/t_textile", "source": "Textile Exchange", "year": 2024,
        },
    },
}

# ---------------------------------------------------------------------------
# Constants - SBTi Pathways
# ---------------------------------------------------------------------------

# Annual linear reduction rates per SBTi pathway
# Source: SBTi Corporate Net-Zero Standard v1.0
# Base year typically 2019/2020, targets to 2030/2050

SBTI_PATHWAYS: Dict[str, Dict[str, Any]] = {
    SBTiPathway.WELL_BELOW_2C: {
        "annual_reduction_rate": 0.025,  # 2.5% per year
        "target_2030_pct": 25.0,         # 25% reduction by 2030 vs base year
        "target_2050_pct": 72.0,         # 72% reduction by 2050
        "description": "Well-below 2 degrees Celsius pathway",
    },
    SBTiPathway.ONE_POINT_FIVE_C: {
        "annual_reduction_rate": 0.042,  # 4.2% per year
        "target_2030_pct": 42.0,         # 42% reduction by 2030 vs base year
        "target_2050_pct": 90.0,         # 90% reduction by 2050
        "description": "1.5 degrees Celsius pathway (Paris-aligned)",
    },
    SBTiPathway.NET_ZERO_2050: {
        "annual_reduction_rate": 0.046,  # 4.6% per year (near-term) + BVCM
        "target_2030_pct": 46.0,         # 46% reduction by 2030 vs base year
        "target_2050_pct": 95.0,         # 95% reduction by 2050
        "description": "Net-Zero by 2050 (SBTi Corporate Net-Zero Standard)",
    },
}

# ---------------------------------------------------------------------------
# Constants - EU ETS Benchmarks
# ---------------------------------------------------------------------------

# EU ETS product benchmarks for free allocation
# Source: Commission Delegated Regulation (EU) 2019/331, updated values
# Values in tCO2/t_product or tCO2/TJ

ETS_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "cement_clinker": {
        "benchmark": 0.766,
        "unit": "tCO2/t_clinker",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,  # 1.5% per year
    },
    "steel_hot_metal": {
        "benchmark": 1.328,
        "unit": "tCO2/t_hot_metal",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "steel_eaf": {
        "benchmark": 0.283,
        "unit": "tCO2/t_steel",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "aluminium_primary": {
        "benchmark": 1.514,
        "unit": "tCO2/t_aluminium",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "float_glass": {
        "benchmark": 0.453,
        "unit": "tCO2/t_glass",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "container_glass": {
        "benchmark": 0.382,
        "unit": "tCO2/t_glass",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "newsprint": {
        "benchmark": 0.298,
        "unit": "tCO2/t_product",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "ammonia": {
        "benchmark": 1.619,
        "unit": "tCO2/t_ammonia",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "nitric_acid": {
        "benchmark": 0.302,
        "unit": "tCO2/t_nitric_acid",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "hydrogen": {
        "benchmark": 8.850,
        "unit": "tCO2/t_hydrogen",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
    "heat": {
        "benchmark": 62.3,
        "unit": "tCO2/TJ",
        "source": "EU ETS Phase IV",
        "year": 2024,
        "annual_reduction_factor": 0.015,
    },
}

# Sub-sector to ETS benchmark product mapping
SUBSECTOR_ETS_MAP: Dict[str, str] = {
    SubSector.CEMENT: "cement_clinker",
    SubSector.STEEL_BOF: "steel_hot_metal",
    SubSector.STEEL_EAF: "steel_eaf",
    SubSector.ALUMINIUM: "aluminium_primary",
    SubSector.GLASS: "float_glass",
    SubSector.PULP_PAPER: "newsprint",
    SubSector.CHEMICALS: "ammonia",
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """Configuration for benchmark assessment."""
    reporting_year: int = Field(description="Reporting year")
    sub_sector: str = Field(description="Manufacturing sub-sector for benchmarking")
    peer_group_size: int = Field(
        default=50, ge=10, le=500,
        description="Assumed peer group size for context",
    )
    include_sbti: bool = Field(default=True, description="Include SBTi alignment assessment")
    include_ets_benchmark: bool = Field(
        default=True, description="Include EU ETS benchmark gap analysis"
    )
    include_trajectory: bool = Field(
        default=True, description="Include multi-year trajectory analysis"
    )
    baseline_year: int = Field(default=2019, description="Baseline year for trajectory")
    target_year: int = Field(default=2030, description="Target year for trajectory")
    sbti_pathway: SBTiPathway = Field(
        default=SBTiPathway.ONE_POINT_FIVE_C,
        description="SBTi pathway for alignment assessment",
    )

    @field_validator("reporting_year", "baseline_year", "target_year", mode="before")
    @classmethod
    def _validate_year(cls, v: Any) -> int:
        year = int(v)
        if year < 2010 or year > 2060:
            raise ValueError(f"Year {year} outside valid range 2010-2060")
        return year

class FacilityKPIs(BaseModel):
    """Facility key performance indicators for benchmarking."""
    facility_id: str = Field(description="Unique facility identifier")
    facility_name: str = Field(default="", description="Facility name")
    sub_sector: str = Field(description="Manufacturing sub-sector")
    emission_intensity_tco2e_per_unit: Decimal = Field(
        description="Emission intensity (tCO2e per production unit)"
    )
    energy_intensity_mj_per_unit: Decimal = Field(
        default=Decimal("0"),
        description="Energy intensity (MJ per production unit)",
    )
    water_intensity_m3_per_unit: Decimal = Field(
        default=Decimal("0"),
        description="Water intensity (m3 per production unit)",
    )
    waste_intensity_kg_per_unit: Decimal = Field(
        default=Decimal("0"),
        description="Waste intensity (kg per production unit)",
    )
    circularity_rate_pct: Decimal = Field(
        default=Decimal("0"),
        description="Circularity rate (%)",
    )
    renewable_share_pct: Decimal = Field(
        default=Decimal("0"),
        description="Renewable energy share (%)",
    )
    scope3_ratio_pct: Decimal = Field(
        default=Decimal("0"),
        description="Scope 3 as percentage of total emissions (%)",
    )
    safety_ltir: Decimal = Field(
        default=Decimal("0"),
        description="Lost Time Injury Rate (per million hours worked)",
    )
    revenue_eur: Decimal = Field(
        default=Decimal("0"),
        description="Annual revenue (EUR)",
    )
    production_volume: Decimal = Field(
        default=Decimal("0"),
        description="Annual production volume (in sector-relevant units)",
    )
    baseline_emission_intensity: Optional[Decimal] = Field(
        default=None,
        description="Baseline emission intensity for trajectory (tCO2e/unit)",
    )

    @field_validator(
        "emission_intensity_tco2e_per_unit", "energy_intensity_mj_per_unit",
        "water_intensity_m3_per_unit", "waste_intensity_kg_per_unit",
        "circularity_rate_pct", "renewable_share_pct", "scope3_ratio_pct",
        "safety_ltir", "revenue_eur", "production_volume",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("baseline_emission_intensity", mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class SectorBenchmark(BaseModel):
    """Sector benchmark data for a specific KPI."""
    sub_sector: str = Field(description="Manufacturing sub-sector")
    kpi: BenchmarkKPI = Field(description="Key performance indicator")
    top_quartile: float = Field(description="Top quartile threshold value")
    median: float = Field(description="Median value")
    bottom_quartile: float = Field(description="Bottom quartile threshold value")
    unit: str = Field(description="Unit of measurement")
    source: str = Field(default="", description="Data source")
    year: int = Field(default=2024, description="Data year")

class KPIRanking(BaseModel):
    """Ranking result for a single KPI."""
    kpi: str = Field(description="KPI name")
    value: float = Field(description="Facility value")
    rank: PercentileRank = Field(description="Quartile ranking")
    top_quartile_threshold: float = Field(description="Top quartile threshold")
    median: float = Field(description="Sector median")
    bottom_quartile_threshold: float = Field(description="Bottom quartile threshold")
    unit: str = Field(description="Unit of measurement")
    gap_to_top_quartile_pct: float = Field(
        default=0.0,
        description="Gap to top quartile (%, positive = above threshold)",
    )

class SBTiAlignment(BaseModel):
    """SBTi pathway alignment assessment."""
    pathway: SBTiPathway = Field(description="SBTi pathway assessed")
    target_year: int = Field(description="Target year")
    required_reduction_pct: float = Field(
        description="Required reduction from baseline (%)"
    )
    current_reduction_pct: float = Field(
        description="Current reduction from baseline (%)"
    )
    on_track: bool = Field(description="Whether facility is on track")
    gap_pct: float = Field(
        default=0.0,
        description="Gap to required reduction (percentage points)",
    )
    annual_reduction_needed: float = Field(
        default=0.0,
        description="Annual reduction needed to meet target (%/year)",
    )
    pathway_description: str = Field(default="", description="Pathway description")

class TrajectoryAnalysis(BaseModel):
    """Multi-year trajectory analysis for emission reduction."""
    baseline_year: int = Field(description="Baseline year")
    baseline_value: float = Field(description="Baseline emission intensity")
    current_year: int = Field(description="Current reporting year")
    current_value: float = Field(description="Current emission intensity")
    target_value: float = Field(description="Target emission intensity")
    target_year: int = Field(description="Target year")
    actual_reduction_pct: float = Field(
        description="Actual reduction from baseline (%)"
    )
    required_reduction_pct: float = Field(
        description="Required reduction from baseline (%)"
    )
    annual_reduction_rate: float = Field(
        description="Actual annual reduction rate (%/year)"
    )
    required_reduction_rate: float = Field(
        description="Required annual reduction rate (%/year)"
    )
    on_track: bool = Field(description="Whether trajectory is on track")
    years_to_target: int = Field(
        default=0, description="Years remaining to target"
    )

class ETSBenchmarkResult(BaseModel):
    """EU ETS benchmark gap analysis result."""
    product_benchmark: str = Field(description="ETS product benchmark name")
    benchmark_value: float = Field(description="ETS benchmark value")
    benchmark_unit: str = Field(description="Unit of measurement")
    facility_value: float = Field(description="Facility emission intensity")
    gap_pct: float = Field(
        description="Gap to benchmark (%, positive = above benchmark)"
    )
    above_benchmark: bool = Field(
        description="Whether facility is above (worse than) benchmark"
    )
    free_allocation_eligible: bool = Field(
        description="Eligible for free allocation at current level"
    )
    annual_reduction_factor: float = Field(
        default=0.015,
        description="Annual reduction factor for benchmark tightening",
    )

class BenchmarkResult(BaseModel):
    """Complete benchmark assessment result with provenance."""
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    facility_id: str = Field(description="Assessed facility ID")
    facility_name: str = Field(default="", description="Facility name")
    sub_sector: str = Field(description="Manufacturing sub-sector")
    # --- Overall ---
    overall_percentile: str = Field(
        default="", description="Overall performance classification"
    )
    kpi_count_assessed: int = Field(default=0, description="Number of KPIs assessed")
    kpi_count_top_quartile: int = Field(
        default=0, description="KPIs in top quartile"
    )
    kpi_count_bottom_quartile: int = Field(
        default=0, description="KPIs in bottom quartile"
    )
    # --- KPI rankings ---
    kpi_rankings: List[KPIRanking] = Field(
        default_factory=list, description="Per-KPI ranking results"
    )
    # --- SBTi ---
    sbti_alignment: Optional[SBTiAlignment] = Field(
        default=None, description="SBTi pathway alignment"
    )
    # --- ETS ---
    ets_benchmark: Optional[ETSBenchmarkResult] = Field(
        default=None, description="EU ETS benchmark analysis"
    )
    ets_benchmark_gap_pct: float = Field(
        default=0.0, description="Gap to EU ETS benchmark (%)"
    )
    # --- Trajectory ---
    trajectory_analysis: Optional[TrajectoryAnalysis] = Field(
        default=None, description="Multi-year trajectory analysis"
    )
    # --- Priorities ---
    improvement_priorities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Prioritized improvement areas"
    )
    # --- Peer comparison ---
    peer_comparison: List[Dict[str, Any]] = Field(
        default_factory=list, description="Peer comparison summary"
    )
    # --- Metadata ---
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology and assumption notes"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ManufacturingBenchmarkEngine:
    """Manufacturing sustainability benchmarking engine.

    Provides deterministic, zero-hallucination benchmarking for:
    - KPI ranking against sector peers (quartile classification)
    - SBTi pathway alignment assessment
    - EU ETS product benchmark gap analysis
    - Multi-year emission reduction trajectory analysis
    - Improvement priority identification

    All benchmark data is sourced from published industry reports.
    Every result includes a SHA-256 provenance hash for audit trails.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the ManufacturingBenchmarkEngine.

        Args:
            config: Configuration including sub-sector, pathway, and
                    feature flags.
        """
        self.config = config
        self._notes: List[str] = []
        logger.info(
            "ManufacturingBenchmarkEngine v%s initialized for sub-sector '%s', year %d",
            _MODULE_VERSION,
            config.sub_sector,
            config.reporting_year,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def benchmark_facility(self, facility: FacilityKPIs) -> BenchmarkResult:
        """Benchmark a facility against sector peers.

        Evaluates all available KPIs, determines quartile rankings,
        and optionally assesses SBTi alignment and ETS benchmark gaps.

        Args:
            facility: Facility KPI data for benchmarking.

        Returns:
            BenchmarkResult with complete assessment and provenance.
        """
        start_time = time.perf_counter()
        self._notes = []

        sub_sector = facility.sub_sector or self.config.sub_sector

        # --- KPI Rankings ---
        kpi_rankings: List[KPIRanking] = []
        top_count = 0
        bottom_count = 0

        kpi_value_map = {
            BenchmarkKPI.EMISSION_INTENSITY: facility.emission_intensity_tco2e_per_unit,
            BenchmarkKPI.ENERGY_INTENSITY: facility.energy_intensity_mj_per_unit,
            BenchmarkKPI.WATER_INTENSITY: facility.water_intensity_m3_per_unit,
            BenchmarkKPI.WASTE_INTENSITY: facility.waste_intensity_kg_per_unit,
            BenchmarkKPI.CIRCULARITY_RATE: facility.circularity_rate_pct,
            BenchmarkKPI.RENEWABLE_SHARE: facility.renewable_share_pct,
            BenchmarkKPI.SCOPE3_RATIO: facility.scope3_ratio_pct,
            BenchmarkKPI.SAFETY_LTIR: facility.safety_ltir,
        }

        for kpi, value in kpi_value_map.items():
            if value == 0 and kpi not in (
                BenchmarkKPI.EMISSION_INTENSITY,
            ):
                continue  # Skip KPIs with zero value (not reported)

            ranking = self._rank_kpi_internal(float(value), sub_sector, kpi)
            if ranking is not None:
                kpi_rankings.append(ranking)
                if ranking.rank == PercentileRank.TOP_QUARTILE:
                    top_count += 1
                elif ranking.rank == PercentileRank.BOTTOM_QUARTILE:
                    bottom_count += 1

        assessed_count = len(kpi_rankings)

        # Overall classification
        if assessed_count == 0:
            overall = "not_assessed"
        elif top_count > assessed_count / 2:
            overall = "leader"
        elif bottom_count > assessed_count / 2:
            overall = "laggard"
        elif top_count >= bottom_count:
            overall = "above_average"
        else:
            overall = "below_average"

        self._notes.append(
            f"Benchmarked {assessed_count} KPIs: {top_count} top quartile, "
            f"{bottom_count} bottom quartile. Overall: {overall}."
        )

        # --- SBTi alignment ---
        sbti_result = None
        if self.config.include_sbti and facility.baseline_emission_intensity is not None:
            sbti_result = self.assess_sbti_alignment(
                facility, self.config.sbti_pathway
            )

        # --- ETS benchmark ---
        ets_result = None
        ets_gap = 0.0
        if self.config.include_ets_benchmark:
            ets_result = self.calculate_ets_gap(facility)
            if ets_result is not None:
                ets_gap = ets_result.gap_pct

        # --- Trajectory analysis ---
        trajectory = None
        if (
            self.config.include_trajectory
            and facility.baseline_emission_intensity is not None
        ):
            trajectory = self.analyze_trajectory(
                baseline_value=float(facility.baseline_emission_intensity),
                current_value=float(facility.emission_intensity_tco2e_per_unit),
                target_reduction_pct=SBTI_PATHWAYS[self.config.sbti_pathway]["target_2030_pct"],
                baseline_year=self.config.baseline_year,
                current_year=self.config.reporting_year,
                target_year=self.config.target_year,
            )

        # --- Improvement priorities ---
        improvement_priorities = self._identify_priorities(kpi_rankings)

        # --- Peer comparison summary ---
        peer_comparison = self._build_peer_comparison(kpi_rankings, sub_sector)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = BenchmarkResult(
            facility_id=facility.facility_id,
            facility_name=facility.facility_name,
            sub_sector=sub_sector,
            overall_percentile=overall,
            kpi_count_assessed=assessed_count,
            kpi_count_top_quartile=top_count,
            kpi_count_bottom_quartile=bottom_count,
            kpi_rankings=kpi_rankings,
            sbti_alignment=sbti_result,
            ets_benchmark=ets_result,
            ets_benchmark_gap_pct=ets_gap,
            trajectory_analysis=trajectory,
            improvement_priorities=improvement_priorities,
            peer_comparison=peer_comparison,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    def rank_kpi(
        self, value: float, sub_sector: str, kpi: BenchmarkKPI
    ) -> PercentileRank:
        """Rank a KPI value against sector benchmarks.

        For intensity metrics (lower is better): below top_quartile = TOP.
        For positive metrics (higher is better, e.g., circularity): above
        top_quartile = TOP.

        Args:
            value: The KPI value to rank.
            sub_sector: Manufacturing sub-sector.
            kpi: The KPI type.

        Returns:
            PercentileRank classification.

        Raises:
            ValueError: If sub-sector or KPI not found in benchmark data.
        """
        ranking = self._rank_kpi_internal(value, sub_sector, kpi)
        if ranking is None:
            raise ValueError(
                f"No benchmark data for sub-sector '{sub_sector}', KPI '{kpi.value}'"
            )
        return ranking.rank

    def assess_sbti_alignment(
        self, facility: FacilityKPIs, pathway: SBTiPathway
    ) -> SBTiAlignment:
        """Assess facility alignment with an SBTi reduction pathway.

        Calculates the required and actual emission reductions from
        baseline, determines on-track status, and computes the annual
        reduction rate needed to close any gap.

        Args:
            facility: Facility KPIs with baseline and current emission intensity.
            pathway: SBTi pathway to assess against.

        Returns:
            SBTiAlignment with on-track status and gap analysis.
        """
        pathway_data = SBTI_PATHWAYS[pathway]
        baseline = float(facility.baseline_emission_intensity or facility.emission_intensity_tco2e_per_unit)
        current = float(facility.emission_intensity_tco2e_per_unit)

        # Years elapsed since baseline
        years_elapsed = self.config.reporting_year - self.config.baseline_year
        years_to_target = self.config.target_year - self.config.reporting_year

        # Required reduction by target year
        if self.config.target_year <= 2030:
            required_reduction_pct = pathway_data["target_2030_pct"]
        else:
            required_reduction_pct = pathway_data["target_2050_pct"]

        # Required reduction by current year (linear interpolation)
        total_years = self.config.target_year - self.config.baseline_year
        if total_years > 0:
            required_by_now_pct = required_reduction_pct * (years_elapsed / total_years)
        else:
            required_by_now_pct = required_reduction_pct

        # Actual reduction
        if baseline > 0:
            actual_reduction_pct = ((baseline - current) / baseline) * 100.0
        else:
            actual_reduction_pct = 0.0

        on_track = actual_reduction_pct >= required_by_now_pct
        gap = round(required_by_now_pct - actual_reduction_pct, 2)

        # Annual reduction needed to close gap
        annual_needed = 0.0
        if years_to_target > 0 and not on_track:
            remaining_reduction_needed = required_reduction_pct - actual_reduction_pct
            if remaining_reduction_needed > 0:
                annual_needed = round(remaining_reduction_needed / years_to_target, 2)

        status_text = "ON TRACK" if on_track else "OFF TRACK"
        self._notes.append(
            f"SBTi {pathway.value}: {status_text}. "
            f"Actual reduction: {round(actual_reduction_pct, 1)}%, "
            f"Required by {self.config.reporting_year}: {round(required_by_now_pct, 1)}%"
        )

        return SBTiAlignment(
            pathway=pathway,
            target_year=self.config.target_year,
            required_reduction_pct=round(required_by_now_pct, 2),
            current_reduction_pct=round(actual_reduction_pct, 2),
            on_track=on_track,
            gap_pct=max(gap, 0.0),
            annual_reduction_needed=annual_needed,
            pathway_description=pathway_data["description"],
        )

    def analyze_trajectory(
        self,
        baseline_value: float,
        current_value: float,
        target_reduction_pct: float,
        baseline_year: int = 2019,
        current_year: int = 2024,
        target_year: int = 2030,
    ) -> TrajectoryAnalysis:
        """Analyze emission reduction trajectory.

        Calculates actual versus required annual reduction rates and
        determines whether the facility is on track to meet its target.

        Args:
            baseline_value: Emission intensity at baseline year.
            current_value: Current emission intensity.
            target_reduction_pct: Required total reduction percentage.
            baseline_year: Year of baseline measurement.
            current_year: Current reporting year.
            target_year: Target year for reduction.

        Returns:
            TrajectoryAnalysis with on-track status and rate comparison.
        """
        years_elapsed = current_year - baseline_year
        years_remaining = target_year - current_year
        total_years = target_year - baseline_year

        # Target value
        target_value = baseline_value * (1.0 - target_reduction_pct / 100.0)

        # Actual reduction
        if baseline_value > 0:
            actual_reduction_pct = ((baseline_value - current_value) / baseline_value) * 100.0
        else:
            actual_reduction_pct = 0.0

        # Annual reduction rate (compound)
        if baseline_value > 0 and current_value > 0 and years_elapsed > 0:
            ratio = current_value / baseline_value
            if ratio > 0:
                annual_rate = (1.0 - math.pow(ratio, 1.0 / years_elapsed)) * 100.0
            else:
                annual_rate = 100.0
        else:
            annual_rate = 0.0

        # Required annual reduction rate from now
        if current_value > 0 and years_remaining > 0 and target_value < current_value:
            remaining_ratio = target_value / current_value
            if remaining_ratio > 0:
                required_rate = (1.0 - math.pow(remaining_ratio, 1.0 / years_remaining)) * 100.0
            else:
                required_rate = 100.0
        elif current_value <= target_value:
            required_rate = 0.0  # Already at or below target
        else:
            required_rate = 0.0

        on_track = current_value <= (
            baseline_value * (1.0 - (target_reduction_pct / 100.0) * (years_elapsed / max(total_years, 1)))
        )

        self._notes.append(
            f"Trajectory: {round(actual_reduction_pct, 1)}% reduced "
            f"({baseline_year}-{current_year}), "
            f"annual rate {round(annual_rate, 2)}%/yr, "
            f"required {round(required_rate, 2)}%/yr to meet {target_year} target."
        )

        return TrajectoryAnalysis(
            baseline_year=baseline_year,
            baseline_value=round(baseline_value, 6),
            current_year=current_year,
            current_value=round(current_value, 6),
            target_value=round(target_value, 6),
            target_year=target_year,
            actual_reduction_pct=round(actual_reduction_pct, 2),
            required_reduction_pct=round(target_reduction_pct, 2),
            annual_reduction_rate=round(annual_rate, 2),
            required_reduction_rate=round(required_rate, 2),
            on_track=on_track,
            years_to_target=max(years_remaining, 0),
        )

    def calculate_ets_gap(
        self, facility: FacilityKPIs
    ) -> Optional[ETSBenchmarkResult]:
        """Calculate gap to EU ETS product benchmark.

        Compares facility emission intensity against the applicable EU ETS
        product benchmark, accounting for annual tightening.

        Args:
            facility: Facility KPI data.

        Returns:
            ETSBenchmarkResult if applicable benchmark found, None otherwise.
        """
        sub_sector = facility.sub_sector or self.config.sub_sector

        # Find applicable ETS benchmark
        ets_key = SUBSECTOR_ETS_MAP.get(sub_sector)
        if ets_key is None or ets_key not in ETS_BENCHMARKS:
            self._notes.append(
                f"No EU ETS product benchmark found for sub-sector '{sub_sector}'."
            )
            return None

        ets_data = ETS_BENCHMARKS[ets_key]
        base_benchmark = ets_data["benchmark"]
        reduction_factor = ets_data["annual_reduction_factor"]
        benchmark_year = ets_data["year"]

        # Adjust benchmark for annual tightening
        years_since = self.config.reporting_year - benchmark_year
        adjusted_benchmark = base_benchmark * (1.0 - reduction_factor * years_since)
        adjusted_benchmark = max(adjusted_benchmark, 0.0)

        facility_value = float(facility.emission_intensity_tco2e_per_unit)

        # Calculate gap
        if adjusted_benchmark > 0:
            gap_pct = ((facility_value - adjusted_benchmark) / adjusted_benchmark) * 100.0
        else:
            gap_pct = 0.0

        above_benchmark = facility_value > adjusted_benchmark

        self._notes.append(
            f"EU ETS benchmark ({ets_key}): {round(adjusted_benchmark, 3)} {ets_data['unit']} "
            f"(adjusted for {self.config.reporting_year}). "
            f"Facility: {round(facility_value, 3)}. "
            f"Gap: {round(gap_pct, 1)}%."
        )

        return ETSBenchmarkResult(
            product_benchmark=ets_key,
            benchmark_value=round(adjusted_benchmark, 4),
            benchmark_unit=ets_data["unit"],
            facility_value=round(facility_value, 4),
            gap_pct=round(gap_pct, 2),
            above_benchmark=above_benchmark,
            free_allocation_eligible=not above_benchmark,
            annual_reduction_factor=reduction_factor,
        )

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _rank_kpi_internal(
        self, value: float, sub_sector: str, kpi: BenchmarkKPI
    ) -> Optional[KPIRanking]:
        """Rank a KPI value against sector benchmarks.

        For 'lower is better' KPIs (intensity metrics, LTIR, scope3_ratio):
            value <= top_quartile  -->  TOP_QUARTILE
            value <= median        -->  SECOND_QUARTILE
            value <= bottom_quartile --> THIRD_QUARTILE
            value > bottom_quartile --> BOTTOM_QUARTILE

        For 'higher is better' KPIs (circularity, renewable_share):
            value >= top_quartile  -->  TOP_QUARTILE
            value >= median        -->  SECOND_QUARTILE
            value >= bottom_quartile --> THIRD_QUARTILE
            value < bottom_quartile --> BOTTOM_QUARTILE

        Args:
            value: The KPI value.
            sub_sector: Sub-sector key.
            kpi: KPI type.

        Returns:
            KPIRanking or None if no benchmark data available.
        """
        if sub_sector not in SECTOR_BENCHMARKS:
            return None

        sector_data = SECTOR_BENCHMARKS[sub_sector]
        if kpi not in sector_data:
            return None

        benchmark = sector_data[kpi]
        tq = benchmark["top_quartile"]
        med = benchmark["median"]
        bq = benchmark["bottom_quartile"]
        unit = benchmark["unit"]

        # Determine if lower or higher is better
        higher_is_better = kpi in (
            BenchmarkKPI.CIRCULARITY_RATE,
            BenchmarkKPI.RENEWABLE_SHARE,
        )

        if higher_is_better:
            # Higher is better: top_quartile > median > bottom_quartile
            if value >= tq:
                rank = PercentileRank.TOP_QUARTILE
            elif value >= med:
                rank = PercentileRank.SECOND_QUARTILE
            elif value >= bq:
                rank = PercentileRank.THIRD_QUARTILE
            else:
                rank = PercentileRank.BOTTOM_QUARTILE

            # Gap to top quartile (how much more needed)
            if tq > 0:
                gap = ((tq - value) / tq) * 100.0
            else:
                gap = 0.0
        else:
            # Lower is better: top_quartile < median < bottom_quartile
            if value <= tq:
                rank = PercentileRank.TOP_QUARTILE
            elif value <= med:
                rank = PercentileRank.SECOND_QUARTILE
            elif value <= bq:
                rank = PercentileRank.THIRD_QUARTILE
            else:
                rank = PercentileRank.BOTTOM_QUARTILE

            # Gap to top quartile (how much reduction needed)
            if tq > 0:
                gap = ((value - tq) / tq) * 100.0
            else:
                gap = 0.0

        return KPIRanking(
            kpi=kpi.value,
            value=round(value, 4),
            rank=rank,
            top_quartile_threshold=tq,
            median=med,
            bottom_quartile_threshold=bq,
            unit=unit,
            gap_to_top_quartile_pct=round(max(gap, 0.0), 2),
        )

    def _identify_priorities(
        self, rankings: List[KPIRanking]
    ) -> List[Dict[str, Any]]:
        """Identify improvement priorities based on KPI rankings.

        Bottom-quartile KPIs are prioritized first, then third-quartile,
        sorted by gap to top quartile.

        Args:
            rankings: List of KPI rankings.

        Returns:
            Prioritized list of improvement areas.
        """
        priority_order = {
            PercentileRank.BOTTOM_QUARTILE: 0,
            PercentileRank.THIRD_QUARTILE: 1,
            PercentileRank.SECOND_QUARTILE: 2,
            PercentileRank.TOP_QUARTILE: 3,
        }

        # Sort by priority (worst first), then by gap (largest first)
        sorted_rankings = sorted(
            rankings,
            key=lambda r: (priority_order.get(r.rank, 99), -r.gap_to_top_quartile_pct),
        )

        priorities: List[Dict[str, Any]] = []
        for rank_idx, kr in enumerate(sorted_rankings, 1):
            if kr.rank in (PercentileRank.TOP_QUARTILE,):
                continue  # No improvement needed for top quartile

            priority_level = "critical" if kr.rank == PercentileRank.BOTTOM_QUARTILE else (
                "high" if kr.rank == PercentileRank.THIRD_QUARTILE else "medium"
            )

            priorities.append({
                "rank": rank_idx,
                "kpi": kr.kpi,
                "current_value": kr.value,
                "target_value": kr.top_quartile_threshold,
                "unit": kr.unit,
                "quartile": kr.rank.value,
                "gap_pct": kr.gap_to_top_quartile_pct,
                "priority": priority_level,
            })

        return priorities

    def _build_peer_comparison(
        self, rankings: List[KPIRanking], sub_sector: str
    ) -> List[Dict[str, Any]]:
        """Build peer comparison summary for reporting.

        Args:
            rankings: List of KPI rankings.
            sub_sector: Manufacturing sub-sector.

        Returns:
            List of peer comparison dicts for each KPI.
        """
        comparison: List[Dict[str, Any]] = []
        for kr in rankings:
            comparison.append({
                "kpi": kr.kpi,
                "facility_value": kr.value,
                "sector_top_quartile": kr.top_quartile_threshold,
                "sector_median": kr.median,
                "sector_bottom_quartile": kr.bottom_quartile_threshold,
                "unit": kr.unit,
                "quartile_rank": kr.rank.value,
                "sub_sector": sub_sector,
            })
        return comparison
