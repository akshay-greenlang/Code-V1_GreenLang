# -*- coding: utf-8 -*-
"""
UtilityBenchmarkEngine - PACK-036 Utility Analysis Engine 7
=============================================================

Benchmarks facility energy performance against industry standards (ENERGY STAR,
CIBSE TM46, ASHRAE 100, LEED, NABERS, BREEAM, DIN 18599, EU EPC) using
deterministic EUI calculations, peer comparison ranking, portfolio analysis,
weather-normalised trend detection, and improvement target setting.

Supports multi-commodity consumption (electricity, natural gas, water, steam,
chilled water), dual-unit output (kBtu/ft2/yr and kWh/m2/yr), source energy
accounting with published conversion factors, and Energy Star score estimation
via percentile-based regression.

EUI Calculation:
    Site EUI    = Total_Site_Energy_kBtu / Gross_Floor_Area_ft2
    Source EUI  = Sum(Commodity_kBtu * Source_Factor) / Gross_Floor_Area_ft2
    kWh/m2/yr   = kBtu/ft2/yr * 3.15459  (unit conversion)

Source Energy Factors (site-to-source):
    Electricity:    2.80  (ENERGY STAR Technical Reference 2023, U.S. national avg)
    Natural Gas:    1.047 (ENERGY STAR Technical Reference 2023, pipeline gas)
    Steam:          1.20  (ENERGY STAR Technical Reference 2023, district steam)
    Chilled Water:  1.04  (ENERGY STAR Technical Reference 2023, district chilled)
    Water:          0.00  (not an energy commodity; tracked for utility cost only)

Unit Conversions:
    1 kWh   = 3.412 kBtu
    1 therm = 100 kBtu
    1 GJ    = 947.817 kBtu
    1 MJ    = 0.947817 kBtu
    1 ft2   = 0.092903 m2
    1 m2    = 10.7639 ft2

Weather Normalisation:
    Weather_Normalised_EUI = Actual_EUI * (Normal_DD / Actual_DD)
    where DD = HDD + CDD (heating and cooling degree-days)

Energy Star Score (simplified percentile model):
    Score = 100 - percentile_rank(source_eui, national_distribution)
    Score >= 75 => eligible for ENERGY STAR certification

Regulatory / Standard References:
    - ENERGY STAR Portfolio Manager Technical Reference (EPA, 2023)
    - U.S. EIA CBECS 2018 (Commercial Building Energy Consumption Survey)
    - CIBSE TM46:2008 Energy Benchmarks
    - ASHRAE Standard 100-2018 Energy Efficiency in Buildings
    - DIN 18599 Energetische Bewertung von Gebauden
    - EU Energy Performance of Buildings Directive 2010/31/EU (EPBD)
    - ISO 52000-1:2017 Energy performance of buildings
    - NABERS Energy (National Australian Built Environment Rating System)
    - BREEAM In-Use International 2024

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Source energy factors from published EPA references
    - Benchmark targets from published standards (CIBSE TM46, CBECS)
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-036 Utility Analysis
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BenchmarkStandard(str, Enum):
    """Benchmark standard frameworks for energy performance comparison.

    ENERGY_STAR:  U.S. EPA ENERGY STAR Portfolio Manager (score 1-100).
    CIBSE_TM46:   UK Chartered Institution of Building Services Engineers TM46.
    DIN_18599:    German standard for energy assessment of buildings.
    EU_EPC:       EU Energy Performance Certificate (A-G rating).
    ASHRAE_100:   ASHRAE Standard 100 - Energy Efficiency in Existing Buildings.
    LEED:         LEED v4.1 O+M Energy Performance prerequisite/credit.
    NABERS:       National Australian Built Environment Rating System.
    BREEAM:       Building Research Establishment Environmental Assessment Method.
    """
    ENERGY_STAR = "energy_star"
    CIBSE_TM46 = "cibse_tm46"
    DIN_18599 = "din_18599"
    EU_EPC = "eu_epc"
    ASHRAE_100 = "ashrae_100"
    LEED = "leed"
    NABERS = "nabers"
    BREEAM = "breeam"

class EUIUnit(str, Enum):
    """Energy Use Intensity measurement units.

    KBTU_FT2_YR:  Thousand BTU per square foot per year (U.S. standard).
    KWH_M2_YR:    Kilowatt-hours per square metre per year (metric standard).
    GJ_M2_YR:     Gigajoules per square metre per year.
    MJ_M2_YR:     Megajoules per square metre per year.
    """
    KBTU_FT2_YR = "kbtu_ft2_yr"
    KWH_M2_YR = "kwh_m2_yr"
    GJ_M2_YR = "gj_m2_yr"
    MJ_M2_YR = "mj_m2_yr"

class BuildingType(str, Enum):
    """Building use type classification for benchmark selection.

    Based on ENERGY STAR Portfolio Manager and CIBSE TM46 building categories.
    """
    OFFICE = "office"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    EDUCATION_K12 = "education_k12"
    EDUCATION_UNIVERSITY = "education_university"
    HEALTHCARE_HOSPITAL = "healthcare_hospital"
    HEALTHCARE_CLINIC = "healthcare_clinic"
    HOTEL = "hotel"
    MULTIFAMILY = "multifamily"
    DATA_CENTER = "data_center"
    INDUSTRIAL_LIGHT = "industrial_light"
    INDUSTRIAL_HEAVY = "industrial_heavy"
    MIXED_USE = "mixed_use"
    SUPERMARKET = "supermarket"
    RESTAURANT = "restaurant"
    WORSHIP = "worship"

class PerformanceQuartile(str, Enum):
    """Performance quartile classification by EUI ranking.

    TOP_25:     Best performing (lowest EUI), 0th-25th percentile.
    SECOND_25:  Good performance, 25th-50th percentile.
    THIRD_25:   Below average, 50th-75th percentile.
    BOTTOM_25:  Worst performing (highest EUI), 75th-100th percentile.
    """
    TOP_25 = "top_25"
    SECOND_25 = "second_25"
    THIRD_25 = "third_25"
    BOTTOM_25 = "bottom_25"

class BenchmarkScope(str, Enum):
    """Energy accounting scope for benchmark comparison.

    SITE:    Energy consumed at the building boundary.
    SOURCE:  Total primary energy incl. generation/transmission losses.
    PRIMARY: Non-renewable primary energy per EN 15603.
    """
    SITE = "site"
    SOURCE = "source"
    PRIMARY = "primary"

class TrendDirection(str, Enum):
    """Direction of performance trend over time.

    IMPROVING:          EUI decreasing (positive improvement).
    DECLINING:          EUI increasing (negative trend).
    STABLE:             EUI change within +/-2% threshold.
    INSUFFICIENT_DATA:  Fewer than 2 data points for trend analysis.
    """
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"

class CommodityType(str, Enum):
    """Utility commodity types tracked in consumption data.

    ELECTRICITY:    Grid electricity (kWh).
    NATURAL_GAS:    Pipeline natural gas (therms/kBtu).
    WATER:          Municipal water supply (gallons/m3).
    STEAM:          District steam (kBtu/lbs).
    CHILLED_WATER:  District chilled water (ton-hours/kBtu).
    """
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    WATER = "water"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"

# ---------------------------------------------------------------------------
# Constants -- Source Energy Factors
# ---------------------------------------------------------------------------

# Site-to-source energy conversion factors.
# Source: ENERGY STAR Portfolio Manager Technical Reference, August 2023.
# These include upstream generation/transmission losses for the U.S.
SOURCE_ENERGY_FACTORS: Dict[CommodityType, Dict[str, Any]] = {
    CommodityType.ELECTRICITY: {
        "factor": Decimal("2.80"),
        "source": "ENERGY STAR Technical Reference 2023, U.S. national avg grid",
    },
    CommodityType.NATURAL_GAS: {
        "factor": Decimal("1.047"),
        "source": "ENERGY STAR Technical Reference 2023, pipeline natural gas",
    },
    CommodityType.STEAM: {
        "factor": Decimal("1.20"),
        "source": "ENERGY STAR Technical Reference 2023, district steam",
    },
    CommodityType.CHILLED_WATER: {
        "factor": Decimal("1.04"),
        "source": "ENERGY STAR Technical Reference 2023, district chilled water",
    },
    CommodityType.WATER: {
        "factor": Decimal("0"),
        "source": "Water is not an energy commodity; tracked for utility cost only",
    },
}
"""Site-to-source energy multipliers from ENERGY STAR Technical Reference."""

# ---------------------------------------------------------------------------
# Constants -- Unit Conversions
# ---------------------------------------------------------------------------

# Energy unit conversions to kBtu.
KBTU_PER_KWH: Decimal = Decimal("3.412")
KBTU_PER_THERM: Decimal = Decimal("100")
KBTU_PER_GJ: Decimal = Decimal("947.817")
KBTU_PER_MJ: Decimal = Decimal("0.947817")

# Area conversions.
FT2_PER_M2: Decimal = Decimal("10.7639")
M2_PER_FT2: Decimal = Decimal("0.092903")

# EUI unit conversion factors (from kBtu/ft2/yr to other units).
# kBtu/ft2/yr -> kWh/m2/yr: divide by 3.412, multiply by 10.7639 = multiply by 3.15459
KBTU_FT2_TO_KWH_M2: Decimal = Decimal("3.15459")
KWH_M2_TO_KBTU_FT2: Decimal = Decimal("0.31700")
KWH_M2_TO_GJ_M2: Decimal = Decimal("0.0036")
KWH_M2_TO_MJ_M2: Decimal = Decimal("3.6")

# ---------------------------------------------------------------------------
# Constants -- ENERGY STAR National Median Source EUI (kBtu/ft2/yr)
# ---------------------------------------------------------------------------

# Source: ENERGY STAR Portfolio Manager, CBECS 2018 data tables.
# Median source EUI for each building type; used for percentile estimation.
ENERGY_STAR_MEDIAN_SOURCE_EUI: Dict[BuildingType, Dict[str, Any]] = {
    BuildingType.OFFICE: {
        "median_eui": Decimal("92.9"),
        "p25_eui": Decimal("65.0"),
        "p75_eui": Decimal("135.0"),
        "source": "CBECS 2018 Table C3, Office buildings",
    },
    BuildingType.RETAIL: {
        "median_eui": Decimal("78.1"),
        "p25_eui": Decimal("50.0"),
        "p75_eui": Decimal("115.0"),
        "source": "CBECS 2018, Retail (other than mall)",
    },
    BuildingType.WAREHOUSE: {
        "median_eui": Decimal("27.0"),
        "p25_eui": Decimal("15.0"),
        "p75_eui": Decimal("48.0"),
        "source": "CBECS 2018, Non-refrigerated warehouse",
    },
    BuildingType.EDUCATION_K12: {
        "median_eui": Decimal("58.5"),
        "p25_eui": Decimal("40.0"),
        "p75_eui": Decimal("85.0"),
        "source": "CBECS 2018, Education (K-12 school)",
    },
    BuildingType.EDUCATION_UNIVERSITY: {
        "median_eui": Decimal("115.0"),
        "p25_eui": Decimal("75.0"),
        "p75_eui": Decimal("165.0"),
        "source": "CBECS 2018, College/University",
    },
    BuildingType.HEALTHCARE_HOSPITAL: {
        "median_eui": Decimal("389.0"),
        "p25_eui": Decimal("280.0"),
        "p75_eui": Decimal("520.0"),
        "source": "CBECS 2018, Hospital/Inpatient health",
    },
    BuildingType.HEALTHCARE_CLINIC: {
        "median_eui": Decimal("79.2"),
        "p25_eui": Decimal("52.0"),
        "p75_eui": Decimal("120.0"),
        "source": "CBECS 2018, Outpatient health care",
    },
    BuildingType.HOTEL: {
        "median_eui": Decimal("95.2"),
        "p25_eui": Decimal("65.0"),
        "p75_eui": Decimal("140.0"),
        "source": "CBECS 2018, Lodging",
    },
    BuildingType.MULTIFAMILY: {
        "median_eui": Decimal("67.0"),
        "p25_eui": Decimal("42.0"),
        "p75_eui": Decimal("105.0"),
        "source": "RECS 2020, Multifamily housing (5+ units)",
    },
    BuildingType.DATA_CENTER: {
        "median_eui": Decimal("925.0"),
        "p25_eui": Decimal("500.0"),
        "p75_eui": Decimal("1400.0"),
        "source": "ENERGY STAR Data Center Technical Reference, 2023",
    },
    BuildingType.INDUSTRIAL_LIGHT: {
        "median_eui": Decimal("95.0"),
        "p25_eui": Decimal("55.0"),
        "p75_eui": Decimal("155.0"),
        "source": "CBECS 2018, Light industrial/manufacturing",
    },
    BuildingType.INDUSTRIAL_HEAVY: {
        "median_eui": Decimal("210.0"),
        "p25_eui": Decimal("130.0"),
        "p75_eui": Decimal("350.0"),
        "source": "CBECS 2018, Heavy industrial/manufacturing",
    },
    BuildingType.MIXED_USE: {
        "median_eui": Decimal("85.0"),
        "p25_eui": Decimal("55.0"),
        "p75_eui": Decimal("130.0"),
        "source": "CBECS 2018, Mixed-use (area-weighted estimate)",
    },
    BuildingType.SUPERMARKET: {
        "median_eui": Decimal("199.0"),
        "p25_eui": Decimal("140.0"),
        "p75_eui": Decimal("280.0"),
        "source": "CBECS 2018, Food sales (grocery/supermarket)",
    },
    BuildingType.RESTAURANT: {
        "median_eui": Decimal("432.0"),
        "p25_eui": Decimal("290.0"),
        "p75_eui": Decimal("610.0"),
        "source": "CBECS 2018, Food service (restaurant)",
    },
    BuildingType.WORSHIP: {
        "median_eui": Decimal("35.5"),
        "p25_eui": Decimal("20.0"),
        "p75_eui": Decimal("60.0"),
        "source": "CBECS 2018, Religious worship",
    },
}
"""National median source EUI by building type from CBECS / ENERGY STAR."""

# ---------------------------------------------------------------------------
# Constants -- CIBSE TM46 Benchmarks (kWh/m2/yr)
# ---------------------------------------------------------------------------

# Source: CIBSE TM46:2008, Energy benchmarks.
# "typical" = median stock, "good_practice" = upper quartile (better performance).
# Columns: electricity (kWh/m2/yr), fossil_fuel (kWh/m2/yr).
CIBSE_TM46_BENCHMARKS: Dict[BuildingType, Dict[str, Any]] = {
    BuildingType.OFFICE: {
        "typical_elec": Decimal("95"),
        "typical_fossil": Decimal("120"),
        "good_practice_elec": Decimal("54"),
        "good_practice_fossil": Decimal("73"),
        "source": "CIBSE TM46:2008, General office (standard)",
    },
    BuildingType.RETAIL: {
        "typical_elec": Decimal("165"),
        "typical_fossil": Decimal("55"),
        "good_practice_elec": Decimal("105"),
        "good_practice_fossil": Decimal("30"),
        "source": "CIBSE TM46:2008, General retail",
    },
    BuildingType.WAREHOUSE: {
        "typical_elec": Decimal("30"),
        "typical_fossil": Decimal("55"),
        "good_practice_elec": Decimal("20"),
        "good_practice_fossil": Decimal("30"),
        "source": "CIBSE TM46:2008, Distribution warehouse",
    },
    BuildingType.EDUCATION_K12: {
        "typical_elec": Decimal("40"),
        "typical_fossil": Decimal("150"),
        "good_practice_elec": Decimal("22"),
        "good_practice_fossil": Decimal("90"),
        "source": "CIBSE TM46:2008, Primary/Secondary school",
    },
    BuildingType.EDUCATION_UNIVERSITY: {
        "typical_elec": Decimal("75"),
        "typical_fossil": Decimal("130"),
        "good_practice_elec": Decimal("50"),
        "good_practice_fossil": Decimal("90"),
        "source": "CIBSE TM46:2008, University campus",
    },
    BuildingType.HEALTHCARE_HOSPITAL: {
        "typical_elec": Decimal("90"),
        "typical_fossil": Decimal("310"),
        "good_practice_elec": Decimal("65"),
        "good_practice_fossil": Decimal("220"),
        "source": "CIBSE TM46:2008, General acute hospital",
    },
    BuildingType.HEALTHCARE_CLINIC: {
        "typical_elec": Decimal("55"),
        "typical_fossil": Decimal("140"),
        "good_practice_elec": Decimal("35"),
        "good_practice_fossil": Decimal("85"),
        "source": "CIBSE TM46:2008, Health centre/clinic",
    },
    BuildingType.HOTEL: {
        "typical_elec": Decimal("105"),
        "typical_fossil": Decimal("200"),
        "good_practice_elec": Decimal("65"),
        "good_practice_fossil": Decimal("120"),
        "source": "CIBSE TM46:2008, Hotel (4-star)",
    },
    BuildingType.MULTIFAMILY: {
        "typical_elec": Decimal("50"),
        "typical_fossil": Decimal("110"),
        "good_practice_elec": Decimal("30"),
        "good_practice_fossil": Decimal("65"),
        "source": "CIBSE TM46:2008, Residential (multi-family block)",
    },
    BuildingType.DATA_CENTER: {
        "typical_elec": Decimal("550"),
        "typical_fossil": Decimal("20"),
        "good_practice_elec": Decimal("350"),
        "good_practice_fossil": Decimal("10"),
        "source": "CIBSE TM46:2008 + ASHRAE 90.4, Data centre",
    },
    BuildingType.INDUSTRIAL_LIGHT: {
        "typical_elec": Decimal("45"),
        "typical_fossil": Decimal("120"),
        "good_practice_elec": Decimal("25"),
        "good_practice_fossil": Decimal("70"),
        "source": "CIBSE TM46:2008, Light manufacturing/workshop",
    },
    BuildingType.INDUSTRIAL_HEAVY: {
        "typical_elec": Decimal("85"),
        "typical_fossil": Decimal("290"),
        "good_practice_elec": Decimal("55"),
        "good_practice_fossil": Decimal("190"),
        "source": "CIBSE TM46:2008, Heavy manufacturing",
    },
    BuildingType.MIXED_USE: {
        "typical_elec": Decimal("80"),
        "typical_fossil": Decimal("110"),
        "good_practice_elec": Decimal("50"),
        "good_practice_fossil": Decimal("65"),
        "source": "CIBSE TM46:2008, Mixed-use (weighted average)",
    },
    BuildingType.SUPERMARKET: {
        "typical_elec": Decimal("400"),
        "typical_fossil": Decimal("80"),
        "good_practice_elec": Decimal("280"),
        "good_practice_fossil": Decimal("50"),
        "source": "CIBSE TM46:2008, Supermarket/hypermarket",
    },
    BuildingType.RESTAURANT: {
        "typical_elec": Decimal("185"),
        "typical_fossil": Decimal("330"),
        "good_practice_elec": Decimal("120"),
        "good_practice_fossil": Decimal("210"),
        "source": "CIBSE TM46:2008, Restaurant",
    },
    BuildingType.WORSHIP: {
        "typical_elec": Decimal("25"),
        "typical_fossil": Decimal("65"),
        "good_practice_elec": Decimal("14"),
        "good_practice_fossil": Decimal("35"),
        "source": "CIBSE TM46:2008, Place of worship",
    },
}
"""CIBSE TM46:2008 energy benchmarks by building type."""

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class FacilityMetrics(BaseModel):
    """Facility profile and physical characteristics for benchmarking.

    Attributes:
        facility_id: Unique facility identifier.
        name: Human-readable facility name.
        building_type: Primary building use type classification.
        gross_floor_area_m2: Gross floor area in square metres.
        gross_floor_area_ft2: Gross floor area in square feet.
        year_built: Year of construction.
        occupancy_pct: Average occupancy percentage (0-100).
        operating_hours_per_week: Weekly operating hours.
        num_workers: Number of regular occupants/workers.
        num_computers: Number of personal computers/workstations.
        has_data_center: Whether facility contains an on-site data centre.
        climate_zone: ASHRAE climate zone (e.g. '4A', '5B').
        location_hdd: Annual heating degree-days (base 65F / 18C).
        location_cdd: Annual cooling degree-days (base 65F / 18C).
    """
    facility_id: str = Field(..., min_length=1, description="Facility identifier")
    name: str = Field(..., min_length=1, description="Facility name")
    building_type: BuildingType = Field(
        default=BuildingType.OFFICE, description="Primary building use type"
    )
    gross_floor_area_m2: float = Field(
        ..., gt=0, description="Gross floor area (m2)"
    )
    gross_floor_area_ft2: float = Field(
        default=0.0, ge=0, description="Gross floor area (ft2)"
    )
    year_built: Optional[int] = Field(
        None, ge=1800, le=2030, description="Construction year"
    )
    occupancy_pct: float = Field(
        default=100.0, ge=0, le=100, description="Average occupancy (%)"
    )
    operating_hours_per_week: float = Field(
        default=50.0, ge=0, le=168, description="Weekly operating hours"
    )
    num_workers: Optional[int] = Field(
        None, ge=0, description="Number of regular workers"
    )
    num_computers: Optional[int] = Field(
        None, ge=0, description="Number of computers/workstations"
    )
    has_data_center: bool = Field(
        default=False, description="On-site data centre present"
    )
    climate_zone: str = Field(
        default="", max_length=10, description="ASHRAE climate zone"
    )
    location_hdd: Optional[float] = Field(
        None, ge=0, description="Annual heating degree-days"
    )
    location_cdd: Optional[float] = Field(
        None, ge=0, description="Annual cooling degree-days"
    )

    @model_validator(mode="after")
    def auto_convert_area(self) -> "FacilityMetrics":
        """Auto-compute ft2 from m2 if ft2 is not supplied."""
        if self.gross_floor_area_ft2 <= 0:
            self.gross_floor_area_ft2 = _round2(
                float(_decimal(self.gross_floor_area_m2) * FT2_PER_M2)
            )
        return self

    @field_validator("gross_floor_area_m2")
    @classmethod
    def validate_floor_area(cls, v: float) -> float:
        """Ensure floor area is within plausible bounds."""
        if v > 5_000_000:
            raise ValueError("Floor area exceeds 5 million m2 sanity check")
        return v

class EnergyConsumption(BaseModel):
    """Annual energy consumption record for a single commodity.

    Attributes:
        commodity: Utility commodity type.
        annual_consumption: Raw annual consumption in native unit.
        unit: Unit of the consumption value (e.g. 'kWh', 'therms', 'kBtu').
        site_energy_kbtu: Consumption converted to site energy kBtu.
        source_factor: Site-to-source energy conversion factor.
        source_energy_kbtu: Source energy after applying conversion factor.
    """
    commodity: CommodityType = Field(
        ..., description="Utility commodity type"
    )
    annual_consumption: float = Field(
        ..., ge=0, description="Raw annual consumption"
    )
    unit: str = Field(
        default="kWh", min_length=1, description="Consumption unit"
    )
    site_energy_kbtu: float = Field(
        default=0.0, ge=0, description="Site energy (kBtu)"
    )
    source_factor: float = Field(
        default=1.0, ge=0, description="Site-to-source conversion factor"
    )
    source_energy_kbtu: float = Field(
        default=0.0, ge=0, description="Source energy (kBtu)"
    )

    @model_validator(mode="after")
    def compute_kbtu(self) -> "EnergyConsumption":
        """Auto-compute site/source kBtu if not provided."""
        if self.site_energy_kbtu <= 0 and self.annual_consumption > 0:
            consumption = _decimal(self.annual_consumption)
            unit_lower = self.unit.lower().strip()
            if unit_lower in ("kwh", "kilowatt-hours", "kilowatt_hours"):
                self.site_energy_kbtu = _round2(float(consumption * KBTU_PER_KWH))
            elif unit_lower in ("therms", "therm"):
                self.site_energy_kbtu = _round2(float(consumption * KBTU_PER_THERM))
            elif unit_lower in ("kbtu", "kilo_btu"):
                self.site_energy_kbtu = _round2(float(consumption))
            elif unit_lower in ("gj", "gigajoules"):
                self.site_energy_kbtu = _round2(float(consumption * KBTU_PER_GJ))
            elif unit_lower in ("mj", "megajoules"):
                self.site_energy_kbtu = _round2(float(consumption * KBTU_PER_MJ))
            else:
                # Default: assume kWh
                self.site_energy_kbtu = _round2(float(consumption * KBTU_PER_KWH))

        # Apply source factor
        factor_entry = SOURCE_ENERGY_FACTORS.get(self.commodity)
        if factor_entry and self.source_factor <= 1.0 and self.commodity != CommodityType.WATER:
            self.source_factor = float(factor_entry["factor"])
        if self.source_energy_kbtu <= 0 and self.site_energy_kbtu > 0:
            self.source_energy_kbtu = _round2(
                float(_decimal(self.site_energy_kbtu) * _decimal(self.source_factor))
            )
        return self

# ---------------------------------------------------------------------------
# Pydantic Models -- Intermediate / Output
# ---------------------------------------------------------------------------

class EUICalculation(BaseModel):
    """Calculated Energy Use Intensity values for a facility.

    Attributes:
        facility_id: Facility that was benchmarked.
        site_eui_kbtu_ft2: Site EUI in kBtu/ft2/yr.
        site_eui_kwh_m2: Site EUI in kWh/m2/yr.
        source_eui_kbtu_ft2: Source EUI in kBtu/ft2/yr.
        source_eui_kwh_m2: Source EUI in kWh/m2/yr.
        scope: Energy accounting scope used.
        weather_normalized: Whether degree-day normalisation was applied.
        total_site_energy_kbtu: Total site energy (kBtu).
        total_source_energy_kbtu: Total source energy (kBtu).
        calculated_at: Timestamp of calculation.
    """
    facility_id: str = Field(..., description="Facility identifier")
    site_eui_kbtu_ft2: float = Field(default=0.0, description="Site EUI (kBtu/ft2/yr)")
    site_eui_kwh_m2: float = Field(default=0.0, description="Site EUI (kWh/m2/yr)")
    source_eui_kbtu_ft2: float = Field(default=0.0, description="Source EUI (kBtu/ft2/yr)")
    source_eui_kwh_m2: float = Field(default=0.0, description="Source EUI (kWh/m2/yr)")
    scope: BenchmarkScope = Field(default=BenchmarkScope.SITE, description="Accounting scope")
    weather_normalized: bool = Field(default=False, description="Weather normalised")
    total_site_energy_kbtu: float = Field(default=0.0, description="Total site energy (kBtu)")
    total_source_energy_kbtu: float = Field(default=0.0, description="Total source energy (kBtu)")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")

class BenchmarkTarget(BaseModel):
    """Benchmark reference target from a specific standard.

    Attributes:
        standard: Which benchmark standard.
        building_type: Building type for the benchmark.
        median_eui: Median (typical) EUI from the standard.
        top_quartile_eui: Top quartile (good practice) EUI.
        bottom_quartile_eui: Bottom quartile (poor) EUI.
        unit: EUI unit for these values.
        source_year: Publication year of the reference data.
        variance_from_median_pct: How the facility compares to median (%).
        performance_label: Descriptive label (e.g. "Above median", "Top quartile").
    """
    standard: BenchmarkStandard = Field(..., description="Benchmark standard")
    building_type: BuildingType = Field(..., description="Building type")
    median_eui: float = Field(..., ge=0, description="Median EUI")
    top_quartile_eui: float = Field(..., ge=0, description="Top quartile EUI")
    bottom_quartile_eui: float = Field(..., ge=0, description="Bottom quartile EUI")
    unit: EUIUnit = Field(default=EUIUnit.KBTU_FT2_YR, description="EUI unit")
    source_year: int = Field(default=2023, ge=2000, description="Reference data year")
    variance_from_median_pct: float = Field(
        default=0.0, description="Variance from median (%)"
    )
    performance_label: str = Field(
        default="", description="Performance classification label"
    )

class EnergyStarScore(BaseModel):
    """ENERGY STAR score estimation for a facility.

    Attributes:
        facility_id: Facility identifier.
        score_1_to_100: Estimated ENERGY STAR score (1=worst, 100=best).
        percentile_rank: Facility percentile within national distribution.
        national_median_eui: National median source EUI for building type.
        predicted_eui: Model-predicted source EUI for the facility.
        actual_source_eui: Actual source EUI for the facility.
        eligible: Whether score >= certification_threshold.
        certification_threshold: Minimum score for ENERGY STAR label (75).
    """
    facility_id: str = Field(..., description="Facility identifier")
    score_1_to_100: int = Field(
        ..., ge=1, le=100, description="ENERGY STAR score (1-100)"
    )
    percentile_rank: float = Field(
        default=0.0, ge=0, le=100, description="Percentile rank (%)"
    )
    national_median_eui: float = Field(
        default=0.0, ge=0, description="National median source EUI (kBtu/ft2/yr)"
    )
    predicted_eui: float = Field(
        default=0.0, ge=0, description="Predicted source EUI"
    )
    actual_source_eui: float = Field(
        default=0.0, ge=0, description="Actual source EUI"
    )
    eligible: bool = Field(
        default=False, description="Eligible for ENERGY STAR certification"
    )
    certification_threshold: int = Field(
        default=75, description="Score threshold for certification"
    )

class PeerComparison(BaseModel):
    """Peer group benchmarking result for a single facility.

    Attributes:
        facility_id: Facility being compared.
        peer_group_size: Number of peer facilities in the comparison group.
        facility_rank: Rank (1 = best / lowest EUI).
        facility_percentile: Percentile (100 = best).
        peer_median_eui: Median EUI of the peer group.
        peer_mean_eui: Mean EUI of the peer group.
        peer_best_eui: Lowest (best) EUI in the peer group.
        peer_worst_eui: Highest (worst) EUI in the peer group.
        quartile: Performance quartile assignment.
    """
    facility_id: str = Field(..., description="Facility identifier")
    peer_group_size: int = Field(..., ge=0, description="Number of peers")
    facility_rank: int = Field(..., ge=1, description="Rank (1 = best)")
    facility_percentile: float = Field(
        default=0.0, ge=0, le=100, description="Percentile (100 = best)"
    )
    peer_median_eui: float = Field(default=0.0, ge=0, description="Peer median EUI")
    peer_mean_eui: float = Field(default=0.0, ge=0, description="Peer mean EUI")
    peer_best_eui: float = Field(default=0.0, ge=0, description="Best peer EUI")
    peer_worst_eui: float = Field(default=0.0, ge=0, description="Worst peer EUI")
    quartile: PerformanceQuartile = Field(
        default=PerformanceQuartile.BOTTOM_25, description="Performance quartile"
    )

class PortfolioRanking(BaseModel):
    """Portfolio-level ranking and statistics across multiple facilities.

    Attributes:
        facilities: List of (facility_id, eui) tuples sorted by EUI ascending.
        best_performer: Facility ID with lowest EUI.
        worst_performer: Facility ID with highest EUI.
        portfolio_avg_eui: Simple average EUI across all facilities.
        portfolio_median_eui: Median EUI.
        spread: Difference between worst and best EUI.
        facility_count: Total number of facilities ranked.
        calculated_at: Timestamp of calculation.
    """
    facilities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Facilities sorted by EUI"
    )
    best_performer: str = Field(default="", description="Best performer facility ID")
    worst_performer: str = Field(default="", description="Worst performer facility ID")
    portfolio_avg_eui: float = Field(default=0.0, ge=0, description="Average EUI")
    portfolio_median_eui: float = Field(default=0.0, ge=0, description="Median EUI")
    spread: float = Field(default=0.0, ge=0, description="EUI spread (max - min)")
    facility_count: int = Field(default=0, ge=0, description="Number of facilities")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")

class TrendAnalysis(BaseModel):
    """Multi-period EUI trend analysis for a single facility.

    Attributes:
        facility_id: Facility being analysed.
        periods: List of period labels (e.g. ['2020', '2021', '2022', '2023']).
        eui_values: Corresponding EUI values for each period.
        trend_direction: Overall trend classification.
        annual_change_pct: Average annual rate of change (%).
        baseline_eui: EUI in the first (baseline) period.
        current_eui: EUI in the most recent period.
        cumulative_change_pct: Total change from baseline to current (%).
    """
    facility_id: str = Field(..., description="Facility identifier")
    periods: List[str] = Field(default_factory=list, description="Period labels")
    eui_values: List[float] = Field(default_factory=list, description="EUI values")
    trend_direction: TrendDirection = Field(
        default=TrendDirection.INSUFFICIENT_DATA, description="Trend direction"
    )
    annual_change_pct: float = Field(
        default=0.0, description="Average annual change (%)"
    )
    baseline_eui: float = Field(default=0.0, ge=0, description="Baseline EUI")
    current_eui: float = Field(default=0.0, ge=0, description="Current EUI")
    cumulative_change_pct: float = Field(
        default=0.0, description="Cumulative change (%)"
    )

class NormalizationFactor(BaseModel):
    """A normalisation factor applied to an EUI calculation.

    Attributes:
        factor_name: Name of the normalisation factor.
        factor_value: Numeric value of the factor.
        source: Source/justification for the factor.
        applied_to: Which EUI metric the factor was applied to.
    """
    factor_name: str = Field(..., description="Normalisation factor name")
    factor_value: float = Field(..., description="Factor value")
    source: str = Field(default="", description="Factor source reference")
    applied_to: str = Field(default="", description="Which metric was adjusted")

class BenchmarkResult(BaseModel):
    """Complete benchmark result for a single facility.

    Combines EUI calculation, ENERGY STAR score, standard comparisons,
    peer comparison, trend analysis, and improvement targets into a
    single provenance-hashed output.

    Attributes:
        facility_id: Facility that was benchmarked.
        eui_calculation: Calculated EUI values.
        energy_star_score: Estimated ENERGY STAR score.
        standard_comparisons: Comparisons against multiple standards.
        peer_comparison: Peer group benchmarking result.
        trend: Multi-period trend analysis (if historical data available).
        improvement_targets: Recommended improvement targets.
        normalisation_factors: Factors applied during calculation.
        provenance_hash: SHA-256 hash for audit trail.
        calculated_at: Timestamp.
        processing_time_ms: Processing duration in milliseconds.
        engine_version: Engine version string.
    """
    facility_id: str = Field(..., description="Facility identifier")
    eui_calculation: Optional[EUICalculation] = Field(
        None, description="EUI calculation"
    )
    energy_star_score: Optional[EnergyStarScore] = Field(
        None, description="ENERGY STAR score"
    )
    standard_comparisons: List[BenchmarkTarget] = Field(
        default_factory=list, description="Standard comparisons"
    )
    peer_comparison: Optional[PeerComparison] = Field(
        None, description="Peer comparison"
    )
    trend: Optional[TrendAnalysis] = Field(
        None, description="Trend analysis"
    )
    improvement_targets: Dict[str, Any] = Field(
        default_factory=dict, description="Improvement targets"
    )
    normalisation_factors: List[NormalizationFactor] = Field(
        default_factory=list, description="Normalisation factors applied"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version"
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class UtilityBenchmarkEngine:
    """Utility performance benchmarking engine.

    Benchmarks facility energy performance against industry standards using
    deterministic EUI calculations, peer comparison ranking, portfolio
    analysis, weather-normalised trend detection, and improvement targeting.

    All numeric calculations use Decimal arithmetic for zero-hallucination
    compliance.  Every output carries a SHA-256 provenance hash for complete
    audit trail traceability.

    Attributes:
        engine_id: Unique engine instance identifier.
        created_at: Engine creation timestamp.

    Example:
        >>> engine = UtilityBenchmarkEngine()
        >>> facility = FacilityMetrics(
        ...     facility_id="FAC-001",
        ...     name="HQ Office Tower",
        ...     building_type=BuildingType.OFFICE,
        ...     gross_floor_area_m2=10000.0,
        ... )
        >>> consumption = [
        ...     EnergyConsumption(commodity=CommodityType.ELECTRICITY,
        ...                       annual_consumption=1_200_000, unit="kWh"),
        ...     EnergyConsumption(commodity=CommodityType.NATURAL_GAS,
        ...                       annual_consumption=4000, unit="therms"),
        ... ]
        >>> eui = engine.calculate_eui(facility, consumption)
        >>> assert eui.site_eui_kbtu_ft2 > 0
    """

    def __init__(self) -> None:
        """Initialise UtilityBenchmarkEngine."""
        self.engine_id: str = _new_uuid()
        self.created_at: datetime = utcnow()
        logger.info(
            "UtilityBenchmarkEngine v%s initialised [engine_id=%s]",
            _MODULE_VERSION,
            self.engine_id,
        )

    # -------------------------------------------------------------------
    # Public: EUI Calculation
    # -------------------------------------------------------------------

    def calculate_eui(
        self,
        facility: FacilityMetrics,
        consumption: List[EnergyConsumption],
    ) -> EUICalculation:
        """Calculate site and source Energy Use Intensity for a facility.

        Site EUI  = Total_Site_Energy_kBtu / Gross_Floor_Area_ft2
        Source EUI = Sum(Commodity_kBtu * Source_Factor) / Gross_Floor_Area_ft2

        Both kBtu/ft2/yr and kWh/m2/yr are computed.

        Args:
            facility: Facility profile with floor area.
            consumption: List of annual commodity consumption records.

        Returns:
            EUICalculation with site and source EUI values.

        Raises:
            ValueError: If consumption list is empty.
        """
        t_start = time.perf_counter()

        if not consumption:
            raise ValueError("At least one consumption record is required")

        area_ft2 = _decimal(facility.gross_floor_area_ft2)
        area_m2 = _decimal(facility.gross_floor_area_m2)

        if area_ft2 <= Decimal("0"):
            area_ft2 = area_m2 * FT2_PER_M2

        # Sum site and source energy across all commodities
        total_site_kbtu = Decimal("0")
        total_source_kbtu = Decimal("0")

        for record in consumption:
            site_kbtu = _decimal(record.site_energy_kbtu)
            source_kbtu = _decimal(record.source_energy_kbtu)
            total_site_kbtu += site_kbtu
            total_source_kbtu += source_kbtu

        # EUI = total energy / floor area
        site_eui_kbtu_ft2 = _safe_divide(total_site_kbtu, area_ft2)
        source_eui_kbtu_ft2 = _safe_divide(total_source_kbtu, area_ft2)

        # Convert to kWh/m2/yr
        site_eui_kwh_m2 = site_eui_kbtu_ft2 * KBTU_FT2_TO_KWH_M2
        source_eui_kwh_m2 = source_eui_kbtu_ft2 * KBTU_FT2_TO_KWH_M2

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "EUI calculated for %s: site=%.2f kBtu/ft2, source=%.2f kBtu/ft2 [%.1fms]",
            facility.facility_id,
            float(site_eui_kbtu_ft2),
            float(source_eui_kbtu_ft2),
            elapsed_ms,
        )

        return EUICalculation(
            facility_id=facility.facility_id,
            site_eui_kbtu_ft2=_round2(float(site_eui_kbtu_ft2)),
            site_eui_kwh_m2=_round2(float(site_eui_kwh_m2)),
            source_eui_kbtu_ft2=_round2(float(source_eui_kbtu_ft2)),
            source_eui_kwh_m2=_round2(float(source_eui_kwh_m2)),
            scope=BenchmarkScope.SOURCE,
            weather_normalized=False,
            total_site_energy_kbtu=_round2(float(total_site_kbtu)),
            total_source_energy_kbtu=_round2(float(total_source_kbtu)),
        )

    # -------------------------------------------------------------------
    # Public: Energy Star Score Estimation
    # -------------------------------------------------------------------

    def estimate_energy_star_score(
        self,
        facility: FacilityMetrics,
        eui: EUICalculation,
    ) -> EnergyStarScore:
        """Estimate ENERGY STAR score using percentile-based regression.

        The simplified model places the facility's source EUI within
        the national distribution for its building type.  A score of 50
        represents the national median; 75+ qualifies for ENERGY STAR
        certification.

        Score = 100 - percentile(source_eui, national_distribution)

        Args:
            facility: Facility profile.
            eui: Previously calculated EUI values.

        Returns:
            EnergyStarScore with estimated score and eligibility.
        """
        t_start = time.perf_counter()

        bt = facility.building_type
        ref = ENERGY_STAR_MEDIAN_SOURCE_EUI.get(bt)

        if ref is None:
            logger.warning(
                "No ENERGY STAR reference data for building type %s; "
                "defaulting to OFFICE",
                bt.value,
            )
            ref = ENERGY_STAR_MEDIAN_SOURCE_EUI[BuildingType.OFFICE]

        median_eui = ref["median_eui"]
        p25_eui = ref["p25_eui"]
        p75_eui = ref["p75_eui"]

        actual_source = _decimal(eui.source_eui_kbtu_ft2)

        # Estimate percentile using log-normal interpolation.
        # The distribution is approximated by 3 known quantiles (p25, p50, p75).
        percentile = self._estimate_percentile(actual_source, p25_eui, median_eui, p75_eui)

        # ENERGY STAR score: higher = better (lower EUI)
        # score = 100 - percentile_of_eui_in_distribution
        score = max(1, min(100, int(_round_val(Decimal("100") - percentile, 0))))

        eligible = score >= 75

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "ENERGY STAR score for %s: %d (percentile=%.1f, eligible=%s) [%.1fms]",
            facility.facility_id,
            score,
            float(percentile),
            eligible,
            elapsed_ms,
        )

        return EnergyStarScore(
            facility_id=facility.facility_id,
            score_1_to_100=score,
            percentile_rank=_round2(float(Decimal("100") - percentile)),
            national_median_eui=_round2(float(median_eui)),
            predicted_eui=_round2(float(median_eui)),
            actual_source_eui=_round2(float(actual_source)),
            eligible=eligible,
            certification_threshold=75,
        )

    # -------------------------------------------------------------------
    # Public: Standard Comparison
    # -------------------------------------------------------------------

    def compare_to_standard(
        self,
        eui: EUICalculation,
        building_type: BuildingType,
        standard: BenchmarkStandard,
    ) -> BenchmarkTarget:
        """Compare facility EUI to a benchmark standard.

        Looks up median and quartile EUI values from the specified standard
        and computes variance from median.

        Args:
            eui: Calculated EUI for the facility.
            building_type: Building type for benchmark lookup.
            standard: Which benchmark standard to compare against.

        Returns:
            BenchmarkTarget with comparison metrics.
        """
        t_start = time.perf_counter()

        if standard == BenchmarkStandard.CIBSE_TM46:
            return self._compare_cibse(eui, building_type)
        elif standard == BenchmarkStandard.ENERGY_STAR:
            return self._compare_energy_star(eui, building_type)
        else:
            # For standards without full data tables, use ENERGY STAR as proxy
            return self._compare_energy_star(eui, building_type, label_standard=standard)

    def _compare_energy_star(
        self,
        eui: EUICalculation,
        building_type: BuildingType,
        label_standard: BenchmarkStandard = BenchmarkStandard.ENERGY_STAR,
    ) -> BenchmarkTarget:
        """Compare against ENERGY STAR national distribution.

        Args:
            eui: Calculated EUI.
            building_type: Building type.
            label_standard: Standard label to use in the output.

        Returns:
            BenchmarkTarget with ENERGY STAR-based comparison.
        """
        ref = ENERGY_STAR_MEDIAN_SOURCE_EUI.get(building_type)
        if ref is None:
            ref = ENERGY_STAR_MEDIAN_SOURCE_EUI[BuildingType.OFFICE]

        median = ref["median_eui"]
        p25 = ref["p25_eui"]
        p75 = ref["p75_eui"]
        actual = _decimal(eui.source_eui_kbtu_ft2)

        variance_pct = _safe_divide(
            (actual - median) * Decimal("100"), median
        )

        label = self._classify_performance(actual, p25, median, p75)

        return BenchmarkTarget(
            standard=label_standard,
            building_type=building_type,
            median_eui=_round2(float(median)),
            top_quartile_eui=_round2(float(p25)),
            bottom_quartile_eui=_round2(float(p75)),
            unit=EUIUnit.KBTU_FT2_YR,
            source_year=2023,
            variance_from_median_pct=_round2(float(variance_pct)),
            performance_label=label,
        )

    def _compare_cibse(
        self,
        eui: EUICalculation,
        building_type: BuildingType,
    ) -> BenchmarkTarget:
        """Compare against CIBSE TM46 benchmarks in kWh/m2/yr.

        CIBSE TM46 provides separate electricity and fossil fuel benchmarks.
        The total typical/good-practice values are used for comparison.

        Args:
            eui: Calculated EUI.
            building_type: Building type.

        Returns:
            BenchmarkTarget with CIBSE TM46 comparison.
        """
        ref = CIBSE_TM46_BENCHMARKS.get(building_type)
        if ref is None:
            ref = CIBSE_TM46_BENCHMARKS[BuildingType.OFFICE]

        typical_total = ref["typical_elec"] + ref["typical_fossil"]
        good_total = ref["good_practice_elec"] + ref["good_practice_fossil"]
        # Bottom quartile estimated as 1.5x typical
        bottom_total = typical_total * Decimal("1.5")

        actual_kwh_m2 = _decimal(eui.site_eui_kwh_m2)

        variance_pct = _safe_divide(
            (actual_kwh_m2 - typical_total) * Decimal("100"),
            typical_total,
        )

        label = self._classify_performance(
            actual_kwh_m2, good_total, typical_total, bottom_total
        )

        return BenchmarkTarget(
            standard=BenchmarkStandard.CIBSE_TM46,
            building_type=building_type,
            median_eui=_round2(float(typical_total)),
            top_quartile_eui=_round2(float(good_total)),
            bottom_quartile_eui=_round2(float(bottom_total)),
            unit=EUIUnit.KWH_M2_YR,
            source_year=2008,
            variance_from_median_pct=_round2(float(variance_pct)),
            performance_label=label,
        )

    # -------------------------------------------------------------------
    # Public: Peer Comparison
    # -------------------------------------------------------------------

    def compare_to_peers(
        self,
        facility: FacilityMetrics,
        facility_eui: float,
        peer_euis: List[Dict[str, Any]],
    ) -> PeerComparison:
        """Compare a facility's EUI against a peer group.

        Peer group EUIs are ranked and the facility is placed within the
        distribution.  Percentile, quartile, and statistical summary are
        computed.

        Args:
            facility: Facility being compared.
            facility_eui: The facility's EUI value (site or source).
            peer_euis: List of dicts with 'facility_id' and 'eui' keys.

        Returns:
            PeerComparison result.

        Raises:
            ValueError: If peer list is empty.
        """
        t_start = time.perf_counter()

        if not peer_euis:
            raise ValueError("Peer group must contain at least one facility")

        # Collect all EUI values including the target facility
        all_euis: List[Decimal] = [_decimal(facility_eui)]
        for peer in peer_euis:
            all_euis.append(_decimal(peer.get("eui", 0)))

        # Sort ascending (lower EUI = better)
        all_euis_sorted = sorted(all_euis)
        facility_val = _decimal(facility_eui)

        # Rank: position in sorted list (1-based, lower EUI = rank 1)
        rank = 1
        for val in all_euis_sorted:
            if val < facility_val:
                rank += 1
            else:
                break

        total = len(all_euis_sorted)

        # Percentile: proportion of peers with EUI >= facility EUI
        worse_count = sum(1 for v in all_euis if v >= facility_val and v != facility_val)
        percentile = _safe_divide(
            _decimal(worse_count) * Decimal("100"),
            _decimal(total - 1) if total > 1 else Decimal("1"),
        )

        # Statistical summary
        eui_floats = [float(v) for v in all_euis]
        median_eui = _decimal(statistics.median(eui_floats))
        mean_eui = _safe_divide(
            sum(all_euis, Decimal("0")), _decimal(total)
        )
        best_eui = min(all_euis)
        worst_eui = max(all_euis)

        # Quartile assignment
        quartile = self._assign_quartile(percentile)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Peer comparison for %s: rank=%d/%d, percentile=%.1f, quartile=%s [%.1fms]",
            facility.facility_id,
            rank,
            total,
            float(percentile),
            quartile.value,
            elapsed_ms,
        )

        return PeerComparison(
            facility_id=facility.facility_id,
            peer_group_size=total,
            facility_rank=rank,
            facility_percentile=_round2(float(percentile)),
            peer_median_eui=_round2(float(median_eui)),
            peer_mean_eui=_round2(float(mean_eui)),
            peer_best_eui=_round2(float(best_eui)),
            peer_worst_eui=_round2(float(worst_eui)),
            quartile=quartile,
        )

    # -------------------------------------------------------------------
    # Public: Portfolio Ranking
    # -------------------------------------------------------------------

    def rank_portfolio(
        self,
        facilities: List[Dict[str, Any]],
    ) -> PortfolioRanking:
        """Rank a portfolio of facilities by EUI.

        Sorts facilities by EUI ascending (lower = better), computes
        portfolio statistics (average, median, spread).

        Args:
            facilities: List of dicts with 'facility_id' and 'eui' keys.

        Returns:
            PortfolioRanking with sorted facilities and summary stats.

        Raises:
            ValueError: If facilities list is empty.
        """
        t_start = time.perf_counter()

        if not facilities:
            raise ValueError("At least one facility is required for portfolio ranking")

        # Sort by EUI ascending
        sorted_facs = sorted(facilities, key=lambda f: f.get("eui", 0))

        eui_values: List[Decimal] = []
        ranked_list: List[Dict[str, Any]] = []

        for idx, fac in enumerate(sorted_facs, 1):
            eui = _decimal(fac.get("eui", 0))
            eui_values.append(eui)
            ranked_list.append({
                "rank": idx,
                "facility_id": fac.get("facility_id", f"facility_{idx}"),
                "eui": _round2(float(eui)),
                "name": fac.get("name", ""),
            })

        n = len(eui_values)
        total = sum(eui_values, Decimal("0"))
        avg_eui = _safe_divide(total, _decimal(n))
        median_eui = _decimal(statistics.median([float(v) for v in eui_values]))
        spread = max(eui_values) - min(eui_values)

        best_id = ranked_list[0]["facility_id"]
        worst_id = ranked_list[-1]["facility_id"]

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Portfolio ranked: %d facilities, avg=%.2f, median=%.2f, spread=%.2f [%.1fms]",
            n,
            float(avg_eui),
            float(median_eui),
            float(spread),
            elapsed_ms,
        )

        return PortfolioRanking(
            facilities=ranked_list,
            best_performer=best_id,
            worst_performer=worst_id,
            portfolio_avg_eui=_round2(float(avg_eui)),
            portfolio_median_eui=_round2(float(median_eui)),
            spread=_round2(float(spread)),
            facility_count=n,
        )

    # -------------------------------------------------------------------
    # Public: Trend Analysis
    # -------------------------------------------------------------------

    def analyze_trend(
        self,
        facility_id: str,
        historical_euis: List[Dict[str, Any]],
    ) -> TrendAnalysis:
        """Analyse EUI trend over multiple periods.

        Computes annual rate of change, cumulative change, and trend
        direction classification.  Requires at least 2 data points.

        Args:
            facility_id: Facility identifier.
            historical_euis: List of dicts with 'period' and 'eui' keys,
                ordered chronologically.

        Returns:
            TrendAnalysis with trend classification and metrics.
        """
        t_start = time.perf_counter()

        if len(historical_euis) < 2:
            logger.warning(
                "Insufficient data for trend analysis on %s (%d points)",
                facility_id,
                len(historical_euis),
            )
            return TrendAnalysis(
                facility_id=facility_id,
                periods=[h.get("period", "") for h in historical_euis],
                eui_values=[float(h.get("eui", 0)) for h in historical_euis],
                trend_direction=TrendDirection.INSUFFICIENT_DATA,
            )

        periods: List[str] = []
        eui_values: List[float] = []

        for entry in historical_euis:
            periods.append(str(entry.get("period", "")))
            eui_values.append(float(entry.get("eui", 0)))

        baseline_eui = _decimal(eui_values[0])
        current_eui = _decimal(eui_values[-1])
        n_periods = len(eui_values) - 1

        # Cumulative change
        cumulative_change = _safe_divide(
            (current_eui - baseline_eui) * Decimal("100"),
            baseline_eui,
        )

        # Average annual change
        annual_change = _safe_divide(cumulative_change, _decimal(n_periods))

        # Trend direction: improving if EUI decreasing
        # Stable threshold: +/- 2% per year
        stability_threshold = Decimal("2")

        if abs(annual_change) <= stability_threshold:
            direction = TrendDirection.STABLE
        elif annual_change < Decimal("0"):
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DECLINING

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Trend analysis for %s: direction=%s, annual=%.2f%%, cumulative=%.2f%% [%.1fms]",
            facility_id,
            direction.value,
            float(annual_change),
            float(cumulative_change),
            elapsed_ms,
        )

        return TrendAnalysis(
            facility_id=facility_id,
            periods=periods,
            eui_values=eui_values,
            trend_direction=direction,
            annual_change_pct=_round2(float(annual_change)),
            baseline_eui=_round2(float(baseline_eui)),
            current_eui=_round2(float(current_eui)),
            cumulative_change_pct=_round2(float(cumulative_change)),
        )

    # -------------------------------------------------------------------
    # Public: Weather Normalisation
    # -------------------------------------------------------------------

    def weather_normalize_eui(
        self,
        eui: EUICalculation,
        actual_hdd: float,
        actual_cdd: float,
        normal_hdd: float,
        normal_cdd: float,
    ) -> EUICalculation:
        """Normalise EUI by weather (degree-day adjustment).

        Adjusts the EUI to what it would be under normal (long-term average)
        weather conditions.  Uses combined heating and cooling degree-days.

        Formula:
            Weather_Normalised_EUI = Actual_EUI * (Normal_DD / Actual_DD)
            where DD = HDD + CDD

        Args:
            eui: Raw EUI calculation to normalise.
            actual_hdd: Actual heating degree-days for the measurement period.
            actual_cdd: Actual cooling degree-days for the measurement period.
            normal_hdd: Long-term normal heating degree-days (e.g. 30-yr avg).
            normal_cdd: Long-term normal cooling degree-days.

        Returns:
            New EUICalculation with weather-normalised values.

        Raises:
            ValueError: If actual degree-days sum to zero.
        """
        t_start = time.perf_counter()

        actual_dd = _decimal(actual_hdd) + _decimal(actual_cdd)
        normal_dd = _decimal(normal_hdd) + _decimal(normal_cdd)

        if actual_dd <= Decimal("0"):
            raise ValueError(
                "Actual degree-days (HDD + CDD) must be positive for "
                "weather normalisation"
            )

        if normal_dd <= Decimal("0"):
            raise ValueError(
                "Normal degree-days (HDD + CDD) must be positive for "
                "weather normalisation"
            )

        ratio = _safe_divide(normal_dd, actual_dd, Decimal("1"))

        norm_site_kbtu = _decimal(eui.site_eui_kbtu_ft2) * ratio
        norm_source_kbtu = _decimal(eui.source_eui_kbtu_ft2) * ratio
        norm_site_kwh = norm_site_kbtu * KBTU_FT2_TO_KWH_M2
        norm_source_kwh = norm_source_kbtu * KBTU_FT2_TO_KWH_M2

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Weather normalisation for %s: ratio=%.4f (normal_dd=%.0f / actual_dd=%.0f) [%.1fms]",
            eui.facility_id,
            float(ratio),
            float(normal_dd),
            float(actual_dd),
            elapsed_ms,
        )

        return EUICalculation(
            facility_id=eui.facility_id,
            site_eui_kbtu_ft2=_round2(float(norm_site_kbtu)),
            site_eui_kwh_m2=_round2(float(norm_site_kwh)),
            source_eui_kbtu_ft2=_round2(float(norm_source_kbtu)),
            source_eui_kwh_m2=_round2(float(norm_source_kwh)),
            scope=eui.scope,
            weather_normalized=True,
            total_site_energy_kbtu=eui.total_site_energy_kbtu,
            total_source_energy_kbtu=eui.total_source_energy_kbtu,
        )

    # -------------------------------------------------------------------
    # Public: Improvement Targets
    # -------------------------------------------------------------------

    def set_improvement_targets(
        self,
        current_eui: float,
        building_type: BuildingType,
        target_quartile: PerformanceQuartile = PerformanceQuartile.TOP_25,
    ) -> Dict[str, Any]:
        """Set improvement targets based on benchmark standards.

        Computes the EUI reduction needed to reach a target performance
        level (quartile) and the corresponding percentage improvement.

        Args:
            current_eui: Current source EUI in kBtu/ft2/yr.
            building_type: Building type for benchmark lookup.
            target_quartile: Target performance level.

        Returns:
            Dict with target_eui, reduction_kbtu, reduction_pct, and
            improvement description.
        """
        t_start = time.perf_counter()

        ref = ENERGY_STAR_MEDIAN_SOURCE_EUI.get(building_type)
        if ref is None:
            ref = ENERGY_STAR_MEDIAN_SOURCE_EUI[BuildingType.OFFICE]

        current = _decimal(current_eui)

        # Select target EUI based on quartile
        target_map: Dict[PerformanceQuartile, Decimal] = {
            PerformanceQuartile.TOP_25: ref["p25_eui"],
            PerformanceQuartile.SECOND_25: ref["median_eui"],
            PerformanceQuartile.THIRD_25: ref["p75_eui"],
            PerformanceQuartile.BOTTOM_25: ref["p75_eui"] * Decimal("1.5"),
        }

        target_eui = target_map.get(target_quartile, ref["p25_eui"])

        reduction = current - target_eui
        reduction_pct = _safe_divide(
            reduction * Decimal("100"), current
        )

        # If already at or below target, no improvement needed
        if reduction <= Decimal("0"):
            improvement_desc = (
                f"Facility already meets {target_quartile.value} target "
                f"({_round2(float(target_eui))} kBtu/ft2/yr)"
            )
            reduction = Decimal("0")
            reduction_pct = Decimal("0")
        else:
            improvement_desc = (
                f"Reduce source EUI by {_round2(float(reduction))} kBtu/ft2/yr "
                f"({_round2(float(reduction_pct))}%) to reach "
                f"{target_quartile.value} ({_round2(float(target_eui))} kBtu/ft2/yr)"
            )

        # Estimated annual energy savings (kBtu per ft2)
        # Convert to kWh/m2/yr for metric savings
        savings_kwh_m2 = reduction * KBTU_FT2_TO_KWH_M2

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Improvement targets for %s: current=%.2f, target=%.2f, reduction=%.2f%% [%.1fms]",
            building_type.value,
            float(current),
            float(target_eui),
            float(reduction_pct),
            elapsed_ms,
        )

        return {
            "current_eui_kbtu_ft2": _round2(float(current)),
            "target_eui_kbtu_ft2": _round2(float(target_eui)),
            "target_quartile": target_quartile.value,
            "reduction_kbtu_ft2": _round2(float(reduction)),
            "reduction_pct": _round2(float(reduction_pct)),
            "savings_kwh_m2_yr": _round2(float(savings_kwh_m2)),
            "building_type": building_type.value,
            "description": improvement_desc,
        }

    # -------------------------------------------------------------------
    # Public: Unit Conversion
    # -------------------------------------------------------------------

    def convert_eui_units(
        self,
        value: float,
        from_unit: EUIUnit,
        to_unit: EUIUnit,
    ) -> Decimal:
        """Convert EUI value between different unit systems.

        Supports conversions among kBtu/ft2/yr, kWh/m2/yr, GJ/m2/yr,
        and MJ/m2/yr.

        Args:
            value: EUI numeric value.
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            Converted value as Decimal.
        """
        if from_unit == to_unit:
            return _decimal(value)

        dec_val = _decimal(value)

        # Step 1: convert to kWh/m2/yr as intermediate
        kwh_m2: Decimal
        if from_unit == EUIUnit.KWH_M2_YR:
            kwh_m2 = dec_val
        elif from_unit == EUIUnit.KBTU_FT2_YR:
            kwh_m2 = dec_val * KBTU_FT2_TO_KWH_M2
        elif from_unit == EUIUnit.GJ_M2_YR:
            kwh_m2 = _safe_divide(dec_val, KWH_M2_TO_GJ_M2, Decimal("0"))
        elif from_unit == EUIUnit.MJ_M2_YR:
            kwh_m2 = _safe_divide(dec_val, KWH_M2_TO_MJ_M2, Decimal("0"))
        else:
            kwh_m2 = dec_val

        # Step 2: convert from kWh/m2/yr to target
        result: Decimal
        if to_unit == EUIUnit.KWH_M2_YR:
            result = kwh_m2
        elif to_unit == EUIUnit.KBTU_FT2_YR:
            result = kwh_m2 * KWH_M2_TO_KBTU_FT2
        elif to_unit == EUIUnit.GJ_M2_YR:
            result = kwh_m2 * KWH_M2_TO_GJ_M2
        elif to_unit == EUIUnit.MJ_M2_YR:
            result = kwh_m2 * KWH_M2_TO_MJ_M2
        else:
            result = kwh_m2

        logger.debug(
            "EUI conversion: %.4f %s -> %.4f %s",
            float(dec_val),
            from_unit.value,
            float(result),
            to_unit.value,
        )

        return result

    # -------------------------------------------------------------------
    # Public: Full Benchmark
    # -------------------------------------------------------------------

    def full_benchmark(
        self,
        facility: FacilityMetrics,
        consumption: List[EnergyConsumption],
        peers: Optional[List[Dict[str, Any]]] = None,
        standards: Optional[List[BenchmarkStandard]] = None,
        historical_euis: Optional[List[Dict[str, Any]]] = None,
    ) -> BenchmarkResult:
        """Run a complete benchmarking analysis for a single facility.

        Orchestrates EUI calculation, ENERGY STAR scoring, standard
        comparisons, peer comparison, trend analysis, and improvement
        target setting into a single provenance-hashed result.

        Args:
            facility: Facility profile.
            consumption: Annual energy consumption records.
            peers: Peer facility EUIs for comparison (optional).
            standards: Benchmark standards to compare against (optional).
            historical_euis: Historical EUI data for trend (optional).

        Returns:
            BenchmarkResult with all analysis components.
        """
        t_start = time.perf_counter()

        logger.info(
            "Starting full benchmark for facility %s (%s)",
            facility.facility_id,
            facility.name,
        )

        # 1. EUI Calculation
        eui = self.calculate_eui(facility, consumption)

        # 2. ENERGY STAR Score
        es_score = self.estimate_energy_star_score(facility, eui)

        # 3. Standard Comparisons
        if standards is None:
            standards = [BenchmarkStandard.ENERGY_STAR, BenchmarkStandard.CIBSE_TM46]

        standard_comparisons: List[BenchmarkTarget] = []
        for std in standards:
            comparison = self.compare_to_standard(eui, facility.building_type, std)
            standard_comparisons.append(comparison)

        # 4. Peer Comparison
        peer_comp: Optional[PeerComparison] = None
        if peers and len(peers) > 0:
            peer_comp = self.compare_to_peers(
                facility, eui.source_eui_kbtu_ft2, peers
            )

        # 5. Trend Analysis
        trend: Optional[TrendAnalysis] = None
        if historical_euis and len(historical_euis) >= 2:
            trend = self.analyze_trend(facility.facility_id, historical_euis)

        # 6. Weather Normalisation (if degree-day data available)
        normalisation_factors: List[NormalizationFactor] = []
        if (
            facility.location_hdd is not None
            and facility.location_cdd is not None
            and facility.location_hdd > 0
        ):
            normalisation_factors.append(
                NormalizationFactor(
                    factor_name="location_hdd",
                    factor_value=facility.location_hdd,
                    source="Facility location profile",
                    applied_to="weather_normalisation",
                )
            )
            normalisation_factors.append(
                NormalizationFactor(
                    factor_name="location_cdd",
                    factor_value=facility.location_cdd if facility.location_cdd else 0.0,
                    source="Facility location profile",
                    applied_to="weather_normalisation",
                )
            )

        # 7. Improvement Targets
        improvement_targets = self.set_improvement_targets(
            eui.source_eui_kbtu_ft2,
            facility.building_type,
            PerformanceQuartile.TOP_25,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Build result (provenance hash computed after assembly)
        result = BenchmarkResult(
            facility_id=facility.facility_id,
            eui_calculation=eui,
            energy_star_score=es_score,
            standard_comparisons=standard_comparisons,
            peer_comparison=peer_comp,
            trend=trend,
            improvement_targets=improvement_targets,
            normalisation_factors=normalisation_factors,
            processing_time_ms=_round2(elapsed_ms),
            engine_version=_MODULE_VERSION,
        )

        # Compute provenance hash
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full benchmark complete for %s: site_eui=%.2f kBtu/ft2, "
            "source_eui=%.2f kBtu/ft2, ES_score=%d, hash=%s [%.1fms]",
            facility.facility_id,
            eui.site_eui_kbtu_ft2,
            eui.source_eui_kbtu_ft2,
            es_score.score_1_to_100,
            result.provenance_hash[:16],
            elapsed_ms,
        )

        return result

    # -------------------------------------------------------------------
    # Internal: Percentile Estimation
    # -------------------------------------------------------------------

    def _estimate_percentile(
        self,
        value: Decimal,
        p25: Decimal,
        p50: Decimal,
        p75: Decimal,
    ) -> Decimal:
        """Estimate the percentile of a value in a log-normal distribution.

        Uses linear interpolation between known quantiles (p25, p50, p75)
        on the log-transformed scale to approximate the percentile.

        For values outside the p25-p75 range, uses extrapolation capped
        at 1st and 99th percentile.

        Args:
            value: The EUI value to place in the distribution.
            p25: 25th percentile EUI value (lower = better).
            p50: 50th percentile (median) EUI value.
            p75: 75th percentile EUI value.

        Returns:
            Estimated percentile (0-100) where lower percentile = lower EUI.
        """
        if value <= Decimal("0"):
            return Decimal("1")

        # Work in log space for log-normal approximation
        try:
            ln_val = _decimal(math.log(float(value)))
            ln_p25 = _decimal(math.log(float(p25))) if p25 > 0 else Decimal("0")
            ln_p50 = _decimal(math.log(float(p50))) if p50 > 0 else Decimal("0")
            ln_p75 = _decimal(math.log(float(p75))) if p75 > 0 else Decimal("0")
        except (ValueError, OverflowError):
            return Decimal("50")

        # Linear interpolation in log space
        if ln_val <= ln_p25:
            # Below p25: extrapolate, clamp to 1
            if ln_p50 == ln_p25:
                return Decimal("1")
            ratio = _safe_divide(ln_p25 - ln_val, ln_p50 - ln_p25, Decimal("0"))
            percentile = Decimal("25") - ratio * Decimal("25")
            return max(Decimal("1"), percentile)

        elif ln_val <= ln_p50:
            # Between p25 and p50
            range_width = ln_p50 - ln_p25
            if range_width == Decimal("0"):
                return Decimal("37")
            fraction = _safe_divide(ln_val - ln_p25, range_width)
            return Decimal("25") + fraction * Decimal("25")

        elif ln_val <= ln_p75:
            # Between p50 and p75
            range_width = ln_p75 - ln_p50
            if range_width == Decimal("0"):
                return Decimal("62")
            fraction = _safe_divide(ln_val - ln_p50, range_width)
            return Decimal("50") + fraction * Decimal("25")

        else:
            # Above p75: extrapolate, clamp to 99
            if ln_p75 == ln_p50:
                return Decimal("99")
            ratio = _safe_divide(ln_val - ln_p75, ln_p75 - ln_p50, Decimal("0"))
            percentile = Decimal("75") + ratio * Decimal("25")
            return min(Decimal("99"), percentile)

    # -------------------------------------------------------------------
    # Internal: Performance Classification
    # -------------------------------------------------------------------

    def _classify_performance(
        self,
        actual: Decimal,
        good_practice: Decimal,
        typical: Decimal,
        poor: Decimal,
    ) -> str:
        """Classify performance against benchmark thresholds.

        Args:
            actual: Facility's actual EUI.
            good_practice: Good practice (top quartile) threshold.
            typical: Typical (median) threshold.
            poor: Poor (bottom quartile) threshold.

        Returns:
            Performance classification label.
        """
        if actual <= good_practice:
            return "Top quartile (excellent)"
        elif actual <= typical:
            return "Above median (good)"
        elif actual <= poor:
            return "Below median (improvement needed)"
        else:
            return "Bottom quartile (significant improvement needed)"

    # -------------------------------------------------------------------
    # Internal: Quartile Assignment
    # -------------------------------------------------------------------

    def _assign_quartile(self, percentile: Decimal) -> PerformanceQuartile:
        """Assign a performance quartile based on percentile.

        Higher percentile means better (more peers with worse performance).

        Args:
            percentile: Facility percentile (0-100, 100 = best).

        Returns:
            PerformanceQuartile classification.
        """
        if percentile >= Decimal("75"):
            return PerformanceQuartile.TOP_25
        elif percentile >= Decimal("50"):
            return PerformanceQuartile.SECOND_25
        elif percentile >= Decimal("25"):
            return PerformanceQuartile.THIRD_25
        else:
            return PerformanceQuartile.BOTTOM_25
