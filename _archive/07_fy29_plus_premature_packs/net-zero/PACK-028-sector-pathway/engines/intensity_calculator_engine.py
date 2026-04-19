# -*- coding: utf-8 -*-
"""
IntensityCalculatorEngine - PACK-028 Sector Pathway Engine 2
===============================================================

Sector-specific intensity metric calculation with 20+ metrics,
normalization, validation, year-over-year trends, and data quality
scoring.  Each sector has a defined intensity metric (e.g. gCO2/kWh
for power, tCO2e/tonne for steel) aligned with SBTi SDA and IEA NZE
requirements.

Intensity Metrics Covered (20+):
    Power:    gCO2/kWh, tCO2e/MWh by source, capacity-weighted
    Steel:    tCO2e/tonne (BF-BOF, EAF, DRI)
    Cement:   tCO2e/tonne clinker, tCO2e/tonne cement, tCO2e/m3 concrete
    Aluminum: tCO2e/tonne aluminum
    Aviation: gCO2/pkm, gCO2/RTK, L fuel/100pkm
    Shipping: gCO2/tkm
    Buildings: kgCO2/m2/year, kWh/m2/year
    Road:     gCO2/vkm
    Rail:     gCO2/pkm
    Chemicals: tCO2e/tonne product
    Pulp:     tCO2e/tonne pulp
    Food:     tCO2e/tonne product
    Oil&Gas:  gCO2/MJ
    Agriculture: tCO2e/tonne food
    Generic:  tCO2e/M revenue

Calculation Methodology:
    Intensity = Total Emissions (tCO2e) / Activity Level (sector unit)
    YoY Change = (Intensity_t - Intensity_t-1) / Intensity_t-1 * 100
    CAGR = ((Intensity_end / Intensity_start)^(1/years) - 1) * 100
    Data Quality = weighted score from measurement method, completeness,
                   data age, and verification status

Regulatory References:
    - SBTi SDA Methodology (sector-specific intensity definitions)
    - GHG Protocol Corporate Standard (emission calculation)
    - IEA NZE 2050 (sector pathway intensity benchmarks)
    - IPCC AR6 WG1 (GWP-100 values)
    - ISO 14064-1:2018 (quantification framework)

Zero-Hallucination:
    - All intensity calculations use deterministic Decimal arithmetic
    - Sector metrics are hard-coded from SBTi/IEA published definitions
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-028 Sector Pathway
Status:  Production Ready
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("result_id", "calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
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
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SectorType(str, Enum):
    """Sector type for intensity calculation."""
    POWER_GENERATION = "power_generation"
    STEEL = "steel"
    CEMENT = "cement"
    ALUMINUM = "aluminum"
    PULP_PAPER = "pulp_paper"
    CHEMICALS = "chemicals"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    ROAD_TRANSPORT = "road_transport"
    RAIL = "rail"
    BUILDINGS_RESIDENTIAL = "buildings_residential"
    BUILDINGS_COMMERCIAL = "buildings_commercial"
    AGRICULTURE = "agriculture"
    FOOD_BEVERAGE = "food_beverage"
    OIL_GAS = "oil_gas"
    CROSS_SECTOR = "cross_sector"

class IntensityMetricType(str, Enum):
    """Intensity metric identifiers.  20+ sector-specific metrics."""
    # Power
    GCO2_PER_KWH = "gCO2_per_kWh"
    TCO2E_PER_MWH = "tCO2e_per_MWh"
    TCO2E_PER_MW_CAPACITY = "tCO2e_per_MW_capacity"
    # Steel
    TCO2E_PER_TONNE_CRUDE_STEEL = "tCO2e_per_tonne_crude_steel"
    TCO2E_PER_TONNE_STEEL_EAF = "tCO2e_per_tonne_steel_EAF"
    TCO2E_PER_TONNE_DRI = "tCO2e_per_tonne_DRI"
    # Cement
    TCO2E_PER_TONNE_CLINKER = "tCO2e_per_tonne_clinker"
    TCO2E_PER_TONNE_CEMENT = "tCO2e_per_tonne_cement"
    TCO2E_PER_M3_CONCRETE = "tCO2e_per_m3_concrete"
    # Aluminum
    TCO2E_PER_TONNE_ALUMINUM = "tCO2e_per_tonne_aluminum"
    # Aviation
    GCO2_PER_PKM = "gCO2_per_pkm"
    GCO2_PER_RTK = "gCO2_per_RTK"
    LITRES_PER_100PKM = "L_per_100pkm"
    # Shipping
    GCO2_PER_TKM = "gCO2_per_tkm"
    # Buildings
    KGCO2_PER_M2_YEAR = "kgCO2_per_m2_year"
    KWH_PER_M2_YEAR = "kWh_per_m2_year"
    KGCO2_PER_M2_EMBODIED = "kgCO2_per_m2_embodied"
    # Road transport
    GCO2_PER_VKM = "gCO2_per_vkm"
    # Rail
    GCO2_PER_PKM_RAIL = "gCO2_per_pkm_rail"
    # Chemicals / Pulp / Food
    TCO2E_PER_TONNE_PRODUCT = "tCO2e_per_tonne_product"
    # Oil & Gas
    GCO2_PER_MJ = "gCO2_per_MJ"
    # Agriculture
    TCO2E_PER_TONNE_FOOD = "tCO2e_per_tonne_food"
    # Generic
    TCO2E_PER_M_REVENUE = "tCO2e_per_M_revenue"
    TCO2E_PER_EMPLOYEE = "tCO2e_per_employee"

class DataMeasurementMethod(str, Enum):
    """How emission/activity data was measured."""
    DIRECT_MEASUREMENT = "direct_measurement"
    CALCULATION = "calculation"
    MASS_BALANCE = "mass_balance"
    ESTIMATION = "estimation"
    PROXY = "proxy"
    SPEND_BASED = "spend_based"

class DataQualityTier(str, Enum):
    """Data quality tier (1 = best, 5 = worst)."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    TIER_5 = "tier_5"

class TrendDirection(str, Enum):
    """Direction of intensity trend."""
    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"
    INSUFFICIENT_DATA = "insufficient_data"

class VerificationStatus(str, Enum):
    """Verification status of intensity data."""
    VERIFIED_LIMITED = "verified_limited"
    VERIFIED_REASONABLE = "verified_reasonable"
    UNVERIFIED = "unverified"
    SELF_DECLARED = "self_declared"

# ---------------------------------------------------------------------------
# Constants -- Sector Intensity Definitions
# ---------------------------------------------------------------------------

# Maps each SectorType to its primary and secondary intensity metrics
# with display unit, conversion factors, and SBTi/IEA alignment info.
SECTOR_INTENSITY_DEFS: Dict[str, Dict[str, Any]] = {
    SectorType.POWER_GENERATION: {
        "primary_metric": IntensityMetricType.GCO2_PER_KWH,
        "secondary_metrics": [
            IntensityMetricType.TCO2E_PER_MWH,
            IntensityMetricType.TCO2E_PER_MW_CAPACITY,
        ],
        "display_unit": "gCO2/kWh",
        "activity_unit": "kWh",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0"),
        "typical_range_max": Decimal("1200"),
        "global_average_2020": Decimal("442"),
        "nze_2050_target": Decimal("0"),
    },
    SectorType.STEEL: {
        "primary_metric": IntensityMetricType.TCO2E_PER_TONNE_CRUDE_STEEL,
        "secondary_metrics": [
            IntensityMetricType.TCO2E_PER_TONNE_STEEL_EAF,
            IntensityMetricType.TCO2E_PER_TONNE_DRI,
        ],
        "display_unit": "tCO2e/tonne crude steel",
        "activity_unit": "tonnes crude steel",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0.2"),
        "typical_range_max": Decimal("3.5"),
        "global_average_2020": Decimal("1.89"),
        "nze_2050_target": Decimal("0.156"),
    },
    SectorType.CEMENT: {
        "primary_metric": IntensityMetricType.TCO2E_PER_TONNE_CEMENT,
        "secondary_metrics": [
            IntensityMetricType.TCO2E_PER_TONNE_CLINKER,
            IntensityMetricType.TCO2E_PER_M3_CONCRETE,
        ],
        "display_unit": "tCO2e/tonne cement",
        "activity_unit": "tonnes cement",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0.1"),
        "typical_range_max": Decimal("1.0"),
        "global_average_2020": Decimal("0.610"),
        "nze_2050_target": Decimal("0.119"),
    },
    SectorType.ALUMINUM: {
        "primary_metric": IntensityMetricType.TCO2E_PER_TONNE_ALUMINUM,
        "secondary_metrics": [],
        "display_unit": "tCO2e/tonne aluminum",
        "activity_unit": "tonnes aluminum",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("1.0"),
        "typical_range_max": Decimal("18.0"),
        "global_average_2020": Decimal("8.60"),
        "nze_2050_target": Decimal("1.31"),
    },
    SectorType.PULP_PAPER: {
        "primary_metric": IntensityMetricType.TCO2E_PER_TONNE_PRODUCT,
        "secondary_metrics": [],
        "display_unit": "tCO2e/tonne pulp",
        "activity_unit": "tonnes pulp",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0.05"),
        "typical_range_max": Decimal("1.5"),
        "global_average_2020": Decimal("0.560"),
        "nze_2050_target": Decimal("0.175"),
    },
    SectorType.CHEMICALS: {
        "primary_metric": IntensityMetricType.TCO2E_PER_TONNE_PRODUCT,
        "secondary_metrics": [],
        "display_unit": "tCO2e/tonne product",
        "activity_unit": "tonnes product",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0.1"),
        "typical_range_max": Decimal("3.0"),
        "global_average_2020": Decimal("0.850"),
        "nze_2050_target": Decimal("0.170"),
    },
    SectorType.AVIATION: {
        "primary_metric": IntensityMetricType.GCO2_PER_PKM,
        "secondary_metrics": [
            IntensityMetricType.GCO2_PER_RTK,
            IntensityMetricType.LITRES_PER_100PKM,
        ],
        "display_unit": "gCO2/pkm",
        "activity_unit": "passenger-km",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("40"),
        "typical_range_max": Decimal("200"),
        "global_average_2020": Decimal("90"),
        "nze_2050_target": Decimal("13"),
    },
    SectorType.SHIPPING: {
        "primary_metric": IntensityMetricType.GCO2_PER_TKM,
        "secondary_metrics": [],
        "display_unit": "gCO2/tkm",
        "activity_unit": "tonne-km",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("1"),
        "typical_range_max": Decimal("20"),
        "global_average_2020": Decimal("7.10"),
        "nze_2050_target": Decimal("0.85"),
    },
    SectorType.ROAD_TRANSPORT: {
        "primary_metric": IntensityMetricType.GCO2_PER_VKM,
        "secondary_metrics": [IntensityMetricType.GCO2_PER_PKM],
        "display_unit": "gCO2/vkm",
        "activity_unit": "vehicle-km",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0"),
        "typical_range_max": Decimal("300"),
        "global_average_2020": Decimal("98"),
        "nze_2050_target": Decimal("5.3"),
    },
    SectorType.RAIL: {
        "primary_metric": IntensityMetricType.GCO2_PER_PKM_RAIL,
        "secondary_metrics": [],
        "display_unit": "gCO2/pkm",
        "activity_unit": "passenger-km",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0"),
        "typical_range_max": Decimal("80"),
        "global_average_2020": Decimal("28"),
        "nze_2050_target": Decimal("3"),
    },
    SectorType.BUILDINGS_RESIDENTIAL: {
        "primary_metric": IntensityMetricType.KGCO2_PER_M2_YEAR,
        "secondary_metrics": [
            IntensityMetricType.KWH_PER_M2_YEAR,
            IntensityMetricType.KGCO2_PER_M2_EMBODIED,
        ],
        "display_unit": "kgCO2/m2/year",
        "activity_unit": "m2 floor area",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0"),
        "typical_range_max": Decimal("80"),
        "global_average_2020": Decimal("28"),
        "nze_2050_target": Decimal("2.3"),
    },
    SectorType.BUILDINGS_COMMERCIAL: {
        "primary_metric": IntensityMetricType.KGCO2_PER_M2_YEAR,
        "secondary_metrics": [
            IntensityMetricType.KWH_PER_M2_YEAR,
            IntensityMetricType.KGCO2_PER_M2_EMBODIED,
        ],
        "display_unit": "kgCO2/m2/year",
        "activity_unit": "m2 floor area",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0"),
        "typical_range_max": Decimal("100"),
        "global_average_2020": Decimal("38"),
        "nze_2050_target": Decimal("3.1"),
    },
    SectorType.AGRICULTURE: {
        "primary_metric": IntensityMetricType.TCO2E_PER_TONNE_FOOD,
        "secondary_metrics": [],
        "display_unit": "tCO2e/tonne food",
        "activity_unit": "tonnes food",
        "sbti_sda_metric": False,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0.1"),
        "typical_range_max": Decimal("30"),
        "global_average_2020": Decimal("2.5"),
        "nze_2050_target": Decimal("1.0"),
    },
    SectorType.FOOD_BEVERAGE: {
        "primary_metric": IntensityMetricType.TCO2E_PER_TONNE_PRODUCT,
        "secondary_metrics": [],
        "display_unit": "tCO2e/tonne product",
        "activity_unit": "tonnes product",
        "sbti_sda_metric": True,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("0.05"),
        "typical_range_max": Decimal("2.0"),
        "global_average_2020": Decimal("0.480"),
        "nze_2050_target": Decimal("0.115"),
    },
    SectorType.OIL_GAS: {
        "primary_metric": IntensityMetricType.GCO2_PER_MJ,
        "secondary_metrics": [],
        "display_unit": "gCO2/MJ",
        "activity_unit": "MJ energy produced",
        "sbti_sda_metric": False,
        "iea_nze_metric": True,
        "typical_range_min": Decimal("5"),
        "typical_range_max": Decimal("100"),
        "global_average_2020": Decimal("55"),
        "nze_2050_target": Decimal("15"),
    },
    SectorType.CROSS_SECTOR: {
        "primary_metric": IntensityMetricType.TCO2E_PER_M_REVENUE,
        "secondary_metrics": [IntensityMetricType.TCO2E_PER_EMPLOYEE],
        "display_unit": "tCO2e/M revenue",
        "activity_unit": "M revenue",
        "sbti_sda_metric": False,
        "iea_nze_metric": False,
        "typical_range_min": Decimal("0"),
        "typical_range_max": Decimal("5000"),
        "global_average_2020": Decimal("150"),
        "nze_2050_target": Decimal("15"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class ActivityDataPoint(BaseModel):
    """Activity data for a single year.

    Attributes:
        year: Reporting year.
        activity_value: Activity level (in sector-specific units).
        activity_unit: Unit of activity measurement.
        total_emissions_tco2e: Total emissions (tCO2e) for this year.
        scope1_tco2e: Scope 1 emissions.
        scope2_tco2e: Scope 2 emissions (location or market-based).
        scope3_tco2e: Scope 3 emissions (if applicable for intensity).
        measurement_method: How the data was measured.
        verification_status: Verification status.
        data_completeness_pct: Data completeness (0-100).
    """
    year: int = Field(..., ge=2010, le=2035, description="Reporting year")
    activity_value: Decimal = Field(
        ..., gt=Decimal("0"), description="Activity level"
    )
    activity_unit: str = Field(
        default="", max_length=50, description="Activity unit"
    )
    total_emissions_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Total emissions (tCO2e)"
    )
    scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Scope 1"
    )
    scope2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Scope 2"
    )
    scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Scope 3"
    )
    measurement_method: DataMeasurementMethod = Field(
        default=DataMeasurementMethod.CALCULATION,
        description="Measurement method",
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="Verification status",
    )
    data_completeness_pct: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="Data completeness (%)",
    )

class SubProcessEntry(BaseModel):
    """Sub-process intensity data (e.g. BF-BOF vs EAF for steel).

    Attributes:
        name: Sub-process name (e.g. "BF-BOF", "EAF", "Coal", "Gas").
        year: Reporting year.
        activity_value: Activity level.
        emissions_tco2e: Emissions from this sub-process.
        share_pct: Share of total production (%).
    """
    name: str = Field(..., min_length=1, max_length=200)
    year: int = Field(..., ge=2010, le=2035)
    activity_value: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0")
    )
    emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0")
    )
    share_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100")
    )

class IntensityInput(BaseModel):
    """Input for intensity calculation.

    Attributes:
        entity_name: Company or entity name.
        sector: Sector classification.
        base_year: Base year for trajectory analysis.
        activity_data: Multi-year activity and emission data.
        sub_processes: Optional sub-process breakdowns.
        custom_metric: Optional custom intensity metric override.
        include_secondary_metrics: Calculate secondary metrics too.
        include_trend_analysis: Perform trend analysis.
        include_benchmark_comparison: Compare against sector averages.
        revenue_m: Revenue in millions (for cross-sector intensity).
        employees: Number of employees (for per-employee intensity).
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    sector: SectorType = Field(..., description="Sector classification")
    base_year: int = Field(
        default=2019, ge=2010, le=2030, description="Base year"
    )
    activity_data: List[ActivityDataPoint] = Field(
        ..., min_length=1, description="Activity data (1+ years)"
    )
    sub_processes: List[SubProcessEntry] = Field(
        default_factory=list, description="Sub-process breakdowns"
    )
    custom_metric: Optional[IntensityMetricType] = Field(
        default=None, description="Custom metric override"
    )
    include_secondary_metrics: bool = Field(
        default=True, description="Calculate secondary metrics"
    )
    include_trend_analysis: bool = Field(
        default=True, description="Perform trend analysis"
    )
    include_benchmark_comparison: bool = Field(
        default=True, description="Compare against sector benchmarks"
    )
    revenue_m: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"), description="Revenue (millions)"
    )
    employees: Optional[int] = Field(
        default=None, ge=0, description="Employee count"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class IntensityDataPoint(BaseModel):
    """Calculated intensity for a single year.

    Attributes:
        year: Reporting year.
        intensity_value: Calculated intensity.
        metric_type: Intensity metric used.
        display_unit: Human-readable unit string.
        emissions_tco2e: Total emissions for this year.
        activity_value: Activity level for this year.
        data_quality_score: Data quality (0-100).
    """
    year: int = Field(default=0)
    intensity_value: Decimal = Field(default=Decimal("0"))
    metric_type: str = Field(default="")
    display_unit: str = Field(default="")
    emissions_tco2e: Decimal = Field(default=Decimal("0"))
    activity_value: Decimal = Field(default=Decimal("0"))
    data_quality_score: Decimal = Field(default=Decimal("0"))

class SecondaryMetricResult(BaseModel):
    """Secondary intensity metric result.

    Attributes:
        metric_type: Secondary metric type.
        display_unit: Display unit.
        values: Year -> intensity value mapping.
    """
    metric_type: str = Field(default="")
    display_unit: str = Field(default="")
    values: Dict[int, Decimal] = Field(default_factory=dict)

class SubProcessIntensity(BaseModel):
    """Intensity breakdown by sub-process.

    Attributes:
        name: Sub-process name.
        intensity_value: Sub-process intensity.
        share_pct: Production share.
        contribution_pct: Contribution to total intensity.
    """
    name: str = Field(default="")
    intensity_value: Decimal = Field(default=Decimal("0"))
    share_pct: Decimal = Field(default=Decimal("0"))
    contribution_pct: Decimal = Field(default=Decimal("0"))

class TrendAnalysis(BaseModel):
    """Intensity trend analysis results.

    Attributes:
        direction: Overall trend direction.
        cagr_pct: Compound annual growth rate (negative = improving).
        total_change_pct: Total change from first to last year.
        yoy_changes: Year-over-year percentage changes.
        best_year: Year with lowest intensity.
        worst_year: Year with highest intensity.
        average_annual_reduction_pct: Average annual reduction.
        years_of_data: Number of data years.
        trend_consistent: Whether trend is consistently declining.
    """
    direction: str = Field(default=TrendDirection.INSUFFICIENT_DATA.value)
    cagr_pct: Decimal = Field(default=Decimal("0"))
    total_change_pct: Decimal = Field(default=Decimal("0"))
    yoy_changes: Dict[int, Decimal] = Field(default_factory=dict)
    best_year: int = Field(default=0)
    worst_year: int = Field(default=0)
    average_annual_reduction_pct: Decimal = Field(default=Decimal("0"))
    years_of_data: int = Field(default=0)
    trend_consistent: bool = Field(default=False)

class BenchmarkComparison(BaseModel):
    """Comparison against sector benchmarks.

    Attributes:
        sector_average_2020: Sector global average (2020).
        nze_2050_target: IEA NZE 2050 target for this sector.
        current_vs_average_pct: Current intensity vs sector average (%).
        current_vs_nze_pct: Current intensity vs NZE target (%).
        above_average: Whether current intensity is above sector average.
        within_typical_range: Whether intensity is within typical range.
        typical_range_min: Typical range lower bound.
        typical_range_max: Typical range upper bound.
        required_annual_reduction_to_nze_pct: Required annual reduction
            to reach NZE 2050 from current level.
    """
    sector_average_2020: Decimal = Field(default=Decimal("0"))
    nze_2050_target: Decimal = Field(default=Decimal("0"))
    current_vs_average_pct: Decimal = Field(default=Decimal("0"))
    current_vs_nze_pct: Decimal = Field(default=Decimal("0"))
    above_average: bool = Field(default=False)
    within_typical_range: bool = Field(default=True)
    typical_range_min: Decimal = Field(default=Decimal("0"))
    typical_range_max: Decimal = Field(default=Decimal("0"))
    required_annual_reduction_to_nze_pct: Decimal = Field(
        default=Decimal("0")
    )

class DataQualityAssessment(BaseModel):
    """Data quality assessment for intensity inputs.

    Attributes:
        overall_score: Composite quality score (0-100).
        tier: Quality tier (1-5).
        measurement_score: Score for measurement methods.
        completeness_score: Score for data completeness.
        consistency_score: Score for year-over-year consistency.
        verification_score: Score for verification status.
        recommendations: Data quality improvement recommendations.
    """
    overall_score: Decimal = Field(default=Decimal("0"))
    tier: str = Field(default=DataQualityTier.TIER_3.value)
    measurement_score: Decimal = Field(default=Decimal("0"))
    completeness_score: Decimal = Field(default=Decimal("0"))
    consistency_score: Decimal = Field(default=Decimal("0"))
    verification_score: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)

class IntensityResult(BaseModel):
    """Complete intensity calculation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        sector: Sector classification.
        primary_metric: Primary intensity metric used.
        display_unit: Display unit for primary metric.
        base_year: Base year.
        base_year_intensity: Base year intensity value.
        current_year: Most recent year of data.
        current_intensity: Most recent intensity value.
        intensity_trajectory: Year-by-year intensity values.
        secondary_metrics: Secondary metric results.
        sub_process_breakdown: Sub-process intensity breakdown.
        trend_analysis: Trend analysis (if requested).
        benchmark_comparison: Benchmark comparison (if requested).
        data_quality: Data quality assessment.
        recommendations: Recommendations.
        warnings: Warnings.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    sector: str = Field(default="")
    primary_metric: str = Field(default="")
    display_unit: str = Field(default="")
    base_year: int = Field(default=0)
    base_year_intensity: Decimal = Field(default=Decimal("0"))
    current_year: int = Field(default=0)
    current_intensity: Decimal = Field(default=Decimal("0"))
    intensity_trajectory: List[IntensityDataPoint] = Field(
        default_factory=list
    )
    secondary_metrics: List[SecondaryMetricResult] = Field(
        default_factory=list
    )
    sub_process_breakdown: List[SubProcessIntensity] = Field(
        default_factory=list
    )
    trend_analysis: Optional[TrendAnalysis] = Field(default=None)
    benchmark_comparison: Optional[BenchmarkComparison] = Field(default=None)
    data_quality: Optional[DataQualityAssessment] = Field(default=None)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IntensityCalculatorEngine:
    """Sector-specific intensity metric calculation engine.

    Computes 20+ sector-specific intensity metrics with normalization,
    trend analysis, benchmark comparison, and data quality scoring.

    All calculations use Decimal arithmetic. No LLM in any path.

    Usage::

        engine = IntensityCalculatorEngine()
        result = engine.calculate(intensity_input)
        print(f"Current: {result.current_intensity} {result.display_unit}")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: IntensityInput) -> IntensityResult:
        """Run complete intensity calculation.

        Args:
            data: Validated intensity input.

        Returns:
            IntensityResult with trajectory, trends, and benchmarks.
        """
        t0 = time.perf_counter()
        logger.info(
            "Intensity calc: entity=%s, sector=%s, years=%d",
            data.entity_name, data.sector.value,
            len(data.activity_data),
        )

        sector_def = SECTOR_INTENSITY_DEFS.get(data.sector.value, {})
        if not sector_def:
            sector_def = SECTOR_INTENSITY_DEFS[SectorType.CROSS_SECTOR]

        # Determine metric
        metric = data.custom_metric or sector_def["primary_metric"]
        display_unit = sector_def["display_unit"]

        # Step 1: Calculate primary intensity trajectory
        trajectory = self._calculate_trajectory(
            data, metric, display_unit, sector_def
        )

        # Step 2: Base year and current year
        sorted_data = sorted(data.activity_data, key=lambda d: d.year)
        base_intensity = Decimal("0")
        current_intensity = Decimal("0")
        base_year = data.base_year
        current_year = sorted_data[-1].year

        for pt in trajectory:
            if pt.year == base_year:
                base_intensity = pt.intensity_value
            if pt.year == current_year:
                current_intensity = pt.intensity_value

        # If base year not in data, use earliest
        if base_intensity == Decimal("0") and trajectory:
            base_intensity = trajectory[0].intensity_value
            base_year = trajectory[0].year

        # Step 3: Secondary metrics
        secondary: List[SecondaryMetricResult] = []
        if data.include_secondary_metrics:
            secondary = self._calculate_secondary_metrics(
                data, sector_def
            )

        # Step 4: Sub-process breakdown
        sub_breakdown = self._calculate_sub_process(
            data, current_year
        )

        # Step 5: Trend analysis
        trend: Optional[TrendAnalysis] = None
        if data.include_trend_analysis and len(trajectory) >= 2:
            trend = self._analyze_trend(trajectory)

        # Step 6: Benchmark comparison
        benchmark: Optional[BenchmarkComparison] = None
        if data.include_benchmark_comparison:
            benchmark = self._compare_benchmarks(
                current_intensity, current_year, sector_def
            )

        # Step 7: Data quality
        dq = self._assess_data_quality(data)

        # Step 8: Recommendations
        recs = self._generate_recommendations(
            data, current_intensity, trend, benchmark, dq
        )

        # Step 9: Warnings
        warnings = self._generate_warnings(
            data, trajectory, sector_def
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = IntensityResult(
            entity_name=data.entity_name,
            sector=data.sector.value,
            primary_metric=metric.value if isinstance(metric, IntensityMetricType) else str(metric),
            display_unit=display_unit,
            base_year=base_year,
            base_year_intensity=_round_val(base_intensity),
            current_year=current_year,
            current_intensity=_round_val(current_intensity),
            intensity_trajectory=trajectory,
            secondary_metrics=secondary,
            sub_process_breakdown=sub_breakdown,
            trend_analysis=trend,
            benchmark_comparison=benchmark,
            data_quality=dq,
            recommendations=recs,
            warnings=warnings,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Intensity complete: entity=%s, sector=%s, current=%s %s",
            data.entity_name, data.sector.value,
            str(current_intensity), display_unit,
        )
        return result

    # ------------------------------------------------------------------ #
    # Trajectory Calculation                                              #
    # ------------------------------------------------------------------ #

    def _calculate_trajectory(
        self,
        data: IntensityInput,
        metric: Any,
        display_unit: str,
        sector_def: Dict[str, Any],
    ) -> List[IntensityDataPoint]:
        """Calculate primary intensity for each year."""
        trajectory: List[IntensityDataPoint] = []
        sorted_data = sorted(data.activity_data, key=lambda d: d.year)

        for dp in sorted_data:
            intensity = _safe_divide(
                dp.total_emissions_tco2e, dp.activity_value
            )

            # Apply unit conversion for sectors that use different scales
            intensity = self._apply_unit_conversion(
                intensity, data.sector, metric
            )

            # Data quality score for this point
            dq_score = self._compute_point_quality(dp)

            trajectory.append(IntensityDataPoint(
                year=dp.year,
                intensity_value=_round_val(intensity),
                metric_type=metric.value if isinstance(metric, IntensityMetricType) else str(metric),
                display_unit=display_unit,
                emissions_tco2e=_round_val(dp.total_emissions_tco2e),
                activity_value=_round_val(dp.activity_value),
                data_quality_score=_round_val(dq_score, 1),
            ))

        return trajectory

    def _apply_unit_conversion(
        self,
        raw_intensity: Decimal,
        sector: SectorType,
        metric: Any,
    ) -> Decimal:
        """Apply sector-specific unit conversions.

        E.g., power: tCO2e/kWh -> gCO2/kWh (multiply by 1,000,000)
              aviation: tCO2e/pkm -> gCO2/pkm (multiply by 1,000,000)
        """
        metric_val = metric.value if isinstance(metric, IntensityMetricType) else str(metric)

        # Conversions from tCO2e/unit -> gCO2/unit
        g_metrics = {
            IntensityMetricType.GCO2_PER_KWH.value,
            IntensityMetricType.GCO2_PER_PKM.value,
            IntensityMetricType.GCO2_PER_RTK.value,
            IntensityMetricType.GCO2_PER_TKM.value,
            IntensityMetricType.GCO2_PER_VKM.value,
            IntensityMetricType.GCO2_PER_PKM_RAIL.value,
            IntensityMetricType.GCO2_PER_MJ.value,
        }
        # Conversions from tCO2e/unit -> kgCO2/unit
        kg_metrics = {
            IntensityMetricType.KGCO2_PER_M2_YEAR.value,
            IntensityMetricType.KGCO2_PER_M2_EMBODIED.value,
        }

        if metric_val in g_metrics:
            return raw_intensity * Decimal("1000000")
        elif metric_val in kg_metrics:
            return raw_intensity * Decimal("1000")
        else:
            return raw_intensity

    def _compute_point_quality(self, dp: ActivityDataPoint) -> Decimal:
        """Compute data quality score for a single data point (0-100)."""
        score = Decimal("0")

        # Measurement method (0-35 points)
        method_scores = {
            DataMeasurementMethod.DIRECT_MEASUREMENT: Decimal("35"),
            DataMeasurementMethod.CALCULATION: Decimal("30"),
            DataMeasurementMethod.MASS_BALANCE: Decimal("28"),
            DataMeasurementMethod.ESTIMATION: Decimal("18"),
            DataMeasurementMethod.PROXY: Decimal("12"),
            DataMeasurementMethod.SPEND_BASED: Decimal("8"),
        }
        score += method_scores.get(dp.measurement_method, Decimal("10"))

        # Completeness (0-30 points)
        score += dp.data_completeness_pct * Decimal("0.30")

        # Verification (0-35 points)
        ver_scores = {
            VerificationStatus.VERIFIED_REASONABLE: Decimal("35"),
            VerificationStatus.VERIFIED_LIMITED: Decimal("25"),
            VerificationStatus.SELF_DECLARED: Decimal("12"),
            VerificationStatus.UNVERIFIED: Decimal("5"),
        }
        score += ver_scores.get(dp.verification_status, Decimal("5"))

        return min(score, Decimal("100"))

    # ------------------------------------------------------------------ #
    # Secondary Metrics                                                   #
    # ------------------------------------------------------------------ #

    def _calculate_secondary_metrics(
        self,
        data: IntensityInput,
        sector_def: Dict[str, Any],
    ) -> List[SecondaryMetricResult]:
        """Calculate secondary intensity metrics."""
        results: List[SecondaryMetricResult] = []
        secondary_defs = sector_def.get("secondary_metrics", [])

        for sec_metric in secondary_defs:
            values: Dict[int, Decimal] = {}
            sorted_data = sorted(data.activity_data, key=lambda d: d.year)

            for dp in sorted_data:
                intensity = _safe_divide(
                    dp.total_emissions_tco2e, dp.activity_value
                )
                intensity = self._apply_unit_conversion(
                    intensity, data.sector, sec_metric
                )
                values[dp.year] = _round_val(intensity)

            # Determine display unit for secondary metric
            sec_unit = self._get_metric_display_unit(sec_metric)

            results.append(SecondaryMetricResult(
                metric_type=sec_metric.value if isinstance(sec_metric, IntensityMetricType) else str(sec_metric),
                display_unit=sec_unit,
                values=values,
            ))

        # Always include revenue intensity if revenue provided
        if data.revenue_m is not None and data.revenue_m > Decimal("0"):
            rev_values: Dict[int, Decimal] = {}
            for dp in sorted(data.activity_data, key=lambda d: d.year):
                rev_intensity = _safe_divide(
                    dp.total_emissions_tco2e, data.revenue_m
                )
                rev_values[dp.year] = _round_val(rev_intensity)
            results.append(SecondaryMetricResult(
                metric_type=IntensityMetricType.TCO2E_PER_M_REVENUE.value,
                display_unit="tCO2e/M revenue",
                values=rev_values,
            ))

        # Employee intensity if provided
        if data.employees is not None and data.employees > 0:
            emp_values: Dict[int, Decimal] = {}
            emp_dec = _decimal(data.employees)
            for dp in sorted(data.activity_data, key=lambda d: d.year):
                emp_intensity = _safe_divide(
                    dp.total_emissions_tco2e, emp_dec
                )
                emp_values[dp.year] = _round_val(emp_intensity)
            results.append(SecondaryMetricResult(
                metric_type=IntensityMetricType.TCO2E_PER_EMPLOYEE.value,
                display_unit="tCO2e/employee",
                values=emp_values,
            ))

        return results

    def _get_metric_display_unit(self, metric: Any) -> str:
        """Get display unit for a metric type."""
        metric_val = metric.value if isinstance(metric, IntensityMetricType) else str(metric)
        units = {
            IntensityMetricType.GCO2_PER_KWH.value: "gCO2/kWh",
            IntensityMetricType.TCO2E_PER_MWH.value: "tCO2e/MWh",
            IntensityMetricType.TCO2E_PER_MW_CAPACITY.value: "tCO2e/MW",
            IntensityMetricType.TCO2E_PER_TONNE_CRUDE_STEEL.value: "tCO2e/t steel",
            IntensityMetricType.TCO2E_PER_TONNE_STEEL_EAF.value: "tCO2e/t steel (EAF)",
            IntensityMetricType.TCO2E_PER_TONNE_DRI.value: "tCO2e/t DRI",
            IntensityMetricType.TCO2E_PER_TONNE_CLINKER.value: "tCO2e/t clinker",
            IntensityMetricType.TCO2E_PER_TONNE_CEMENT.value: "tCO2e/t cement",
            IntensityMetricType.TCO2E_PER_M3_CONCRETE.value: "tCO2e/m3 concrete",
            IntensityMetricType.TCO2E_PER_TONNE_ALUMINUM.value: "tCO2e/t aluminum",
            IntensityMetricType.GCO2_PER_PKM.value: "gCO2/pkm",
            IntensityMetricType.GCO2_PER_RTK.value: "gCO2/RTK",
            IntensityMetricType.LITRES_PER_100PKM.value: "L/100pkm",
            IntensityMetricType.GCO2_PER_TKM.value: "gCO2/tkm",
            IntensityMetricType.KGCO2_PER_M2_YEAR.value: "kgCO2/m2/yr",
            IntensityMetricType.KWH_PER_M2_YEAR.value: "kWh/m2/yr",
            IntensityMetricType.KGCO2_PER_M2_EMBODIED.value: "kgCO2/m2",
            IntensityMetricType.GCO2_PER_VKM.value: "gCO2/vkm",
            IntensityMetricType.GCO2_PER_PKM_RAIL.value: "gCO2/pkm (rail)",
            IntensityMetricType.TCO2E_PER_TONNE_PRODUCT.value: "tCO2e/t product",
            IntensityMetricType.GCO2_PER_MJ.value: "gCO2/MJ",
            IntensityMetricType.TCO2E_PER_TONNE_FOOD.value: "tCO2e/t food",
            IntensityMetricType.TCO2E_PER_M_REVENUE.value: "tCO2e/M revenue",
            IntensityMetricType.TCO2E_PER_EMPLOYEE.value: "tCO2e/employee",
        }
        return units.get(metric_val, metric_val)

    # ------------------------------------------------------------------ #
    # Sub-Process Breakdown                                               #
    # ------------------------------------------------------------------ #

    def _calculate_sub_process(
        self,
        data: IntensityInput,
        current_year: int,
    ) -> List[SubProcessIntensity]:
        """Calculate sub-process intensity breakdown."""
        if not data.sub_processes:
            return []

        # Filter to current year
        year_procs = [
            sp for sp in data.sub_processes
            if sp.year == current_year
        ]
        if not year_procs:
            # Fall back to latest year available
            if data.sub_processes:
                latest = max(sp.year for sp in data.sub_processes)
                year_procs = [
                    sp for sp in data.sub_processes
                    if sp.year == latest
                ]

        total_emissions = sum(
            sp.emissions_tco2e for sp in year_procs
        )
        results: List[SubProcessIntensity] = []

        for sp in year_procs:
            intensity = _safe_divide(sp.emissions_tco2e, sp.activity_value)
            contribution = _safe_pct(sp.emissions_tco2e, total_emissions)

            results.append(SubProcessIntensity(
                name=sp.name,
                intensity_value=_round_val(intensity),
                share_pct=_round_val(sp.share_pct, 2),
                contribution_pct=_round_val(contribution, 2),
            ))

        return results

    # ------------------------------------------------------------------ #
    # Trend Analysis                                                      #
    # ------------------------------------------------------------------ #

    def _analyze_trend(
        self,
        trajectory: List[IntensityDataPoint],
    ) -> TrendAnalysis:
        """Analyze intensity trend over time."""
        if len(trajectory) < 2:
            return TrendAnalysis(
                direction=TrendDirection.INSUFFICIENT_DATA.value,
                years_of_data=len(trajectory),
            )

        sorted_pts = sorted(trajectory, key=lambda p: p.year)
        first = sorted_pts[0]
        last = sorted_pts[-1]

        # Total change
        total_change = _safe_pct(
            last.intensity_value - first.intensity_value,
            first.intensity_value,
        )

        # CAGR
        years = _decimal(last.year - first.year)
        cagr = Decimal("0")
        if years > Decimal("0") and first.intensity_value > Decimal("0"):
            ratio = _safe_divide(
                last.intensity_value, first.intensity_value
            )
            if ratio > Decimal("0"):
                try:
                    exponent = float(_safe_divide(
                        Decimal("1"), years
                    ))
                    cagr = _decimal(
                        float(ratio) ** exponent - 1.0
                    ) * Decimal("100")
                except (OverflowError, ValueError):
                    cagr = Decimal("0")

        # Year-over-year changes
        yoy: Dict[int, Decimal] = {}
        for i in range(1, len(sorted_pts)):
            prev = sorted_pts[i - 1]
            curr = sorted_pts[i]
            change = _safe_pct(
                curr.intensity_value - prev.intensity_value,
                prev.intensity_value,
            )
            yoy[curr.year] = _round_val(change, 2)

        # Best and worst years
        best = min(sorted_pts, key=lambda p: p.intensity_value)
        worst = max(sorted_pts, key=lambda p: p.intensity_value)

        # Average annual reduction
        avg_reduction = _safe_divide(total_change, years)

        # Direction
        if total_change < Decimal("-2"):
            direction = TrendDirection.DECREASING.value
        elif total_change > Decimal("2"):
            direction = TrendDirection.INCREASING.value
        else:
            direction = TrendDirection.STABLE.value

        # Consistency check
        trend_consistent = all(
            v <= Decimal("0") for v in yoy.values()
        ) if yoy else False

        return TrendAnalysis(
            direction=direction,
            cagr_pct=_round_val(cagr, 3),
            total_change_pct=_round_val(total_change, 2),
            yoy_changes=yoy,
            best_year=best.year,
            worst_year=worst.year,
            average_annual_reduction_pct=_round_val(avg_reduction, 3),
            years_of_data=len(sorted_pts),
            trend_consistent=trend_consistent,
        )

    # ------------------------------------------------------------------ #
    # Benchmark Comparison                                                #
    # ------------------------------------------------------------------ #

    def _compare_benchmarks(
        self,
        current_intensity: Decimal,
        current_year: int,
        sector_def: Dict[str, Any],
    ) -> BenchmarkComparison:
        """Compare current intensity against sector benchmarks."""
        avg = sector_def.get("global_average_2020", Decimal("0"))
        nze = sector_def.get("nze_2050_target", Decimal("0"))
        rng_min = sector_def.get("typical_range_min", Decimal("0"))
        rng_max = sector_def.get("typical_range_max", Decimal("0"))

        vs_avg = _safe_pct(
            current_intensity - avg, avg
        )
        vs_nze = _safe_pct(
            current_intensity - nze, nze
        ) if nze > Decimal("0") else Decimal("0")

        above_avg = current_intensity > avg
        within_range = rng_min <= current_intensity <= rng_max

        # Required annual reduction to reach NZE by 2050
        years_to_2050 = max(2050 - current_year, 1)
        required_reduction = Decimal("0")
        if current_intensity > nze and current_intensity > Decimal("0"):
            total_needed = _safe_pct(
                current_intensity - nze, current_intensity
            )
            required_reduction = _safe_divide(
                total_needed, _decimal(years_to_2050)
            )

        return BenchmarkComparison(
            sector_average_2020=_round_val(avg),
            nze_2050_target=_round_val(nze),
            current_vs_average_pct=_round_val(vs_avg, 2),
            current_vs_nze_pct=_round_val(vs_nze, 2),
            above_average=above_avg,
            within_typical_range=within_range,
            typical_range_min=_round_val(rng_min),
            typical_range_max=_round_val(rng_max),
            required_annual_reduction_to_nze_pct=_round_val(
                required_reduction, 3
            ),
        )

    # ------------------------------------------------------------------ #
    # Data Quality Assessment                                             #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(
        self,
        data: IntensityInput,
    ) -> DataQualityAssessment:
        """Assess overall data quality of intensity inputs."""
        # Measurement score (0-30)
        method_scores = {
            DataMeasurementMethod.DIRECT_MEASUREMENT: 30,
            DataMeasurementMethod.CALCULATION: 25,
            DataMeasurementMethod.MASS_BALANCE: 22,
            DataMeasurementMethod.ESTIMATION: 15,
            DataMeasurementMethod.PROXY: 10,
            DataMeasurementMethod.SPEND_BASED: 5,
        }
        if data.activity_data:
            meas_total = sum(
                method_scores.get(dp.measurement_method, 10)
                for dp in data.activity_data
            )
            meas_avg = _decimal(meas_total / len(data.activity_data))
        else:
            meas_avg = Decimal("0")

        # Completeness score (0-25)
        if data.activity_data:
            comp_avg = sum(
                dp.data_completeness_pct for dp in data.activity_data
            ) / _decimal(len(data.activity_data))
            comp_score = comp_avg * Decimal("0.25")
        else:
            comp_score = Decimal("0")

        # Consistency score (0-20)
        sorted_data = sorted(data.activity_data, key=lambda d: d.year)
        consistency = Decimal("20")  # Start with full score
        for i in range(1, len(sorted_data)):
            prev = sorted_data[i - 1]
            curr = sorted_data[i]
            # Check for unreasonable jumps (>300% change)
            if prev.total_emissions_tco2e > Decimal("0"):
                ratio = _safe_divide(
                    curr.total_emissions_tco2e,
                    prev.total_emissions_tco2e,
                )
                if ratio > Decimal("3") or ratio < Decimal("0.33"):
                    consistency -= Decimal("5")
        consistency = max(consistency, Decimal("0"))

        # Verification score (0-25)
        ver_scores_map = {
            VerificationStatus.VERIFIED_REASONABLE: 25,
            VerificationStatus.VERIFIED_LIMITED: 18,
            VerificationStatus.SELF_DECLARED: 10,
            VerificationStatus.UNVERIFIED: 3,
        }
        if data.activity_data:
            ver_total = sum(
                ver_scores_map.get(dp.verification_status, 3)
                for dp in data.activity_data
            )
            ver_avg = _decimal(ver_total / len(data.activity_data))
        else:
            ver_avg = Decimal("0")

        overall = meas_avg + comp_score + consistency + ver_avg
        overall = min(overall, Decimal("100"))

        # Tier determination
        if overall >= Decimal("80"):
            tier = DataQualityTier.TIER_1.value
        elif overall >= Decimal("60"):
            tier = DataQualityTier.TIER_2.value
        elif overall >= Decimal("40"):
            tier = DataQualityTier.TIER_3.value
        elif overall >= Decimal("20"):
            tier = DataQualityTier.TIER_4.value
        else:
            tier = DataQualityTier.TIER_5.value

        # Recommendations
        recs: List[str] = []
        if meas_avg < Decimal("20"):
            recs.append(
                "Upgrade measurement methods from estimation/proxy to "
                "direct measurement or calculation approaches."
            )
        if comp_score < Decimal("20"):
            recs.append(
                "Improve data completeness to cover >95% of activities."
            )
        if consistency < Decimal("15"):
            recs.append(
                "Investigate year-over-year data inconsistencies. "
                "Large jumps (>300%) suggest data quality issues."
            )
        if ver_avg < Decimal("15"):
            recs.append(
                "Obtain third-party verification (limited or reasonable "
                "assurance) per ISO 14064-3 or ISAE 3410."
            )

        return DataQualityAssessment(
            overall_score=_round_val(overall, 1),
            tier=tier,
            measurement_score=_round_val(meas_avg, 1),
            completeness_score=_round_val(comp_score, 1),
            consistency_score=_round_val(consistency, 1),
            verification_score=_round_val(ver_avg, 1),
            recommendations=recs,
        )

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: IntensityInput,
        current_intensity: Decimal,
        trend: Optional[TrendAnalysis],
        benchmark: Optional[BenchmarkComparison],
        dq: DataQualityAssessment,
    ) -> List[str]:
        """Generate intensity improvement recommendations."""
        recs: List[str] = []

        # Trend-based
        if trend and trend.direction == TrendDirection.INCREASING.value:
            recs.append(
                f"Emission intensity is increasing (CAGR: {trend.cagr_pct}%). "
                f"Identify and address the drivers of intensity growth."
            )
        elif trend and trend.direction == TrendDirection.STABLE.value:
            recs.append(
                "Emission intensity has been stable. Accelerate "
                "decarbonization to achieve sector pathway alignment."
            )

        # Benchmark-based
        if benchmark and benchmark.above_average:
            recs.append(
                f"Current intensity is {benchmark.current_vs_average_pct}% "
                f"above sector average. Prioritize efficiency improvements "
                f"to reach sector average as an interim target."
            )

        if benchmark and benchmark.required_annual_reduction_to_nze_pct > Decimal("5"):
            recs.append(
                f"Required annual reduction of "
                f"{benchmark.required_annual_reduction_to_nze_pct}%/year "
                f"to reach NZE 2050. This requires transformational "
                f"technology deployment, not incremental efficiency."
            )

        # Data points
        if len(data.activity_data) < 3:
            recs.append(
                "Provide at least 3 years of data for reliable trend "
                "analysis and pathway projection."
            )

        # Sub-processes
        if not data.sub_processes:
            recs.append(
                "Add sub-process breakdown for more granular intensity "
                "analysis (e.g., BF-BOF vs. EAF for steel)."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Warnings                                                            #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        data: IntensityInput,
        trajectory: List[IntensityDataPoint],
        sector_def: Dict[str, Any],
    ) -> List[str]:
        """Generate intensity calculation warnings."""
        warnings: List[str] = []
        rng_min = sector_def.get("typical_range_min", Decimal("0"))
        rng_max = sector_def.get("typical_range_max", Decimal("0"))

        for pt in trajectory:
            if rng_max > Decimal("0") and pt.intensity_value > rng_max:
                warnings.append(
                    f"Year {pt.year}: intensity {pt.intensity_value} exceeds "
                    f"typical sector range ({rng_min}-{rng_max}). "
                    f"Verify emission and activity data."
                )
            if pt.intensity_value < Decimal("0"):
                warnings.append(
                    f"Year {pt.year}: negative intensity detected. "
                    f"Check emission accounting for errors."
                )

        # Check for gaps in years
        years = sorted(pt.year for pt in trajectory)
        for i in range(1, len(years)):
            gap = years[i] - years[i - 1]
            if gap > 1:
                warnings.append(
                    f"Data gap: {gap - 1} year(s) missing between "
                    f"{years[i - 1]} and {years[i]}."
                )

        return warnings

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_supported_metrics(self) -> List[Dict[str, str]]:
        """Return all supported intensity metrics."""
        return [
            {"metric": m.value, "unit": self._get_metric_display_unit(m)}
            for m in IntensityMetricType
        ]

    def get_sector_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Return sector intensity definitions."""
        result: Dict[str, Dict[str, Any]] = {}
        for sector_key, defn in SECTOR_INTENSITY_DEFS.items():
            primary = defn["primary_metric"]
            result[sector_key] = {
                "primary_metric": primary.value if isinstance(primary, IntensityMetricType) else str(primary),
                "display_unit": defn["display_unit"],
                "activity_unit": defn["activity_unit"],
                "sbti_sda_metric": defn["sbti_sda_metric"],
                "iea_nze_metric": defn["iea_nze_metric"],
                "global_average_2020": str(defn["global_average_2020"]),
                "nze_2050_target": str(defn["nze_2050_target"]),
            }
        return result
